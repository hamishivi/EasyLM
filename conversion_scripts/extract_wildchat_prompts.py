import json
from copy import deepcopy
from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("allenai/WildChat", split="train")
    def filter_dataset(example):
        if example["language"] != "English":
            return False
        # if example["model"] != "gpt-4":
        #     return False
        if example["conversation"][0]["role"] != "user":
            return False
        if len(example["conversation"][0]["content"].strip().split()) <= 2:
            return False
        if len(example["conversation"][0]["content"].strip().split()) > 500: # these long examples cannot be used in our current PPO training setup
            return False
        if len(example["conversation"][0]["content"].strip()) > 5000:  # there might be examples with a lot of characters concatenated together
            return False
        if example["conversation"][0]["content"].strip().startswith("As a prompt generator for a generative AI called"): # this style of prompt duplicates a lot
            return False
        return True

    filtered_dataset = dataset.filter(filter_dataset, num_proc=64)
    print("Filtered dataset size: ", len(filtered_dataset))

    filtered_gpt4_dataset = filtered_dataset.filter(lambda x: x["model"] == "gpt-4", num_proc=64)
    print("Filtered GPT-4 dataset size: ", len(filtered_gpt4_dataset))

    filtered_chatgpt_dataset = filtered_dataset.filter(lambda x: x["model"] == "gpt-3.5-turbo", num_proc=64)
    print("Filtered ChatGPT dataset size: ", len(filtered_chatgpt_dataset))

    target_num_examples = 60908
    selected_examples = []
    selected_prompts = set()

    for dataset in [filtered_gpt4_dataset, filtered_chatgpt_dataset]:
        dataset = dataset.shuffle(seed=42)
        for example in dataset:
            if len(selected_examples) == target_num_examples:
                break
            if example["conversation"][0]["content"].strip() in selected_prompts:
                continue
            selected_prompts.add(example["conversation"][0]["content"].strip())
            selected_examples.append(example)
        print(len(selected_examples))

    with open("data/wildchat_1turn_60k.jsonl", "w") as f:
        for example in selected_examples:
            new_example = deepcopy(example)
            new_example["conversation"] = [example["conversation"][0], example["conversation"][1]]
            new_example.pop("timestamp")
            f.write(json.dumps(new_example) + "\n")
    
