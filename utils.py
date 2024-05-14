import datasets 
import accelerate
import transformers
import torch
import pandas as pd

### Helper function for reducing size of the training data

def take_a_percentage_of_data(dataset, percentage=0.1, shuffle=True, random_state=None):
    # sort and group the dataset by label
    df = pd.DataFrame(dataset)
    df_sorted = df.sort_values(by='label')
    grouped_dfs = df_sorted.groupby('label')

    # ensure that proportions of the groups remains the same as in the original dataset
    filtered_dfs_per_group = []
    for label, group in grouped_dfs:
        num_samples_to_keep = int(len(group) * percentage)
        filtered_group = group.head(num_samples_to_keep)
        filtered_dfs_per_group.append(filtered_group)

    # concatenate (and shuffle) the filtered group-wise dataframes
    filtered_df = pd.concat(filtered_dfs_per_group)
    if shuffle:
        filtered_df = filtered_df.sample(frac=1, random_state=random_state)

    filtered_df.reset_index(drop=True, inplace=True) # resets the index of the DataFrame, drops the previous index column
    filtered_df_as_dict = filtered_df.to_dict(orient='list')
    filtered_dataset = datasets.Dataset.from_dict(filtered_df_as_dict)
    return filtered_dataset

### Helper function for freezing the pre-trained layers of the adapted GPT2-Models

def freeze_pretrained_layers(network):

    for param in network.transformer.parameters():
        param.requires_grad = False

    for param in network.score.parameters():
        param.requires_grad = True

    for block in network.transformer.h:
        for param in block.attn.parameters():
            param.requires_grad = True

    return network

### Helper function for predicting the label of a sample

def predict_label(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(-1).item()

    return prediction

### Helper function for computing the accuracy achieved on a dataset.

def calculate_accuracy(model, data, tokenizer):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in data:
            prediction = predict_label(item['text'], model, tokenizer)
            label = item['label']
            total += 1
            if label == prediction:
              correct += 1
            if label != 2:
                print("Heureka")
            if label == 2:
                print("oops")

    return correct / total

def compute_accuracy(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}