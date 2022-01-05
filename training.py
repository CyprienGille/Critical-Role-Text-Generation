import torch
import torch.nn as nn
import math
import time
from cr_proc import CRprocessing
from cr_gen import CRTransformer


# use the gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# should be replaced to wherever the text files are
data_dir_path = "crT_text_files/"

crp = CRprocessing(data_dir_path)


def batchify(data, nb_batches):
    """
    Splits data into nb_batches, adding a new dimension
    Note: trims any remainder data that wouldn't fit
    
    Args:
        data {torch.Tensor} -- the data to be spread
        nb_batches {int} -- the size of the batch dimension
    """
    nbatch = data.size(0) // nb_batches
    data = data.narrow(0, 0, nbatch * nb_batches)  # trimming
    data = data.view(nb_batches, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i):
    """returns a sequence (called "batch" during training) for all data batches (that are along dim 0) 
    Also returns an offset-1 sequence for output-target loss computation
    
    Arguments:
        source {slicable} -- [the data to get the batch from]
        i {int} -- [starting pos of the batch]
    """
    seq_len = min(max_len, len(source) - 2 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target


batch_size = 30
eval_batch_size = 10

train_data = batchify(crp.train_data_encoded, batch_size)
val_data = batchify(crp.eval_data_encoded, eval_batch_size)
test_data = batchify(crp.test_data_encoded, eval_batch_size)

max_len = 45  # max length looked at at once by the model

voc_size = len(crp.vocab.stoi)
input_dim = 256
nb_heads = 2
dim_ff = 512
n_enc_layers = 5
n_dec_layers = 5
dropout = 0.1
model = CRTransformer(
    voc_size, input_dim, nb_heads, dim_ff, n_enc_layers, n_dec_layers, dropout
).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # stochastic gradient descent

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=1, gamma=0.3
)  # each step_size epochs: new_lr = old_lr * gamma


def train(logging_interval=200):
    """
    Args:
        logging_interval {int} --  the number of batches to process before printing a log to stdout (default:200)
    """
    # get initiating decoder targets, does not influence the training phase
    model.eval()
    init_data, init_targets = get_batch(train_data, 0)
    prev_outputs = torch.argmax(
        model(init_data, init_targets, src_masked=True, tgt_masked=True), dim=-1
    )

    model.train()  # Turn on the train mode
    total_loss = 0.0
    start_time = time.time()
    for batch_num, i in enumerate(
        range(0, train_data.size(0) - 1 - max_len, max_len)
    ):  # max_len step
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()  # we don't want to backpropagate to the start of the dataset

        output = model(data, prev_outputs, src_masked=True, tgt_masked=True)

        loss = criterion(
            output.view(-1, voc_size), targets.reshape(-1)
        )  # flattened versions to compute loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 0.5
        )  # prevent gradient explosion
        optimizer.step()

        prev_outputs = torch.argmax(
            output.clone().detach(), dim=-1
        )  # detach to avoid autograd intervention

        total_loss += loss.item()
        if batch_num % logging_interval == 0 and batch_num > 0:
            cur_loss = total_loss / logging_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f} | ppl {:8.4f}".format(
                    epoch,
                    batch_num,
                    (len(train_data) // max_len) - 1,
                    elapsed * 1000 / logging_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_for_eval):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        init_data, init_targets = get_batch(data_for_eval, 0)
        prev_outputs = torch.argmax(
            eval_model(init_data, init_targets, src_masked=True, tgt_masked=True),
            dim=-1,
        )
        for i in range(0, data_for_eval.size(0) - 1 - max_len, max_len):
            data, targets = get_batch(data_for_eval, i)
            output = eval_model(data, prev_outputs, src_masked=True, tgt_masked=True)
            prev_outputs = torch.argmax(output.clone().detach(), dim=-1)
            output_flat = output.view(-1, voc_size)
            total_loss += len(data) * criterion(output_flat, targets.reshape(-1)).item()
    return total_loss / (len(data_for_eval) - 1 - max_len)


best_val_loss = float("inf")
epochs = 3  # The number of epochs
best_model = None

try:
    for epoch in range(1, epochs + 1):
        print("-" * 89)
        print("|Start of epoch {:3d}|".format(epoch))
        print("-" * 89)
        epoch_start_time = time.time()
        train(400)
        val_loss = evaluate(model, val_data)
        print("-" * 89)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}".format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
            )
        )
        print("-" * 89)

        if val_loss < best_val_loss:  # keeping the best model ever seen
            best_val_loss = val_loss
            best_model = model

        scheduler.step()  # stepLR

except KeyboardInterrupt:  # exit out of training with a Ctrl+C, code will not fail unless we did not complete epoch 1
    print("-" * 89)
    print("Exiting from training early")


test_loss = evaluate(best_model, test_data)
print("=" * 89)
print(
    "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
        test_loss, math.exp(test_loss)
    )
)
print("=" * 89)

torch.save(
    best_model.state_dict(), "best_model.pt"
)  # saving the model to reuse it for inference later

