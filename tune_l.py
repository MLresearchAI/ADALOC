# coding:utf8

import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import evaluate
import numpy as np
import pandas as pd

def main(args):
    datasets = ["qnli", "sst2", "tweet_eval"]
    p_values = [1, 0.05, 0.10]  
    results = []

    for dataset_name in datasets:
        metric = evaluate.load("glue", dataset_name) if dataset_name in ["mrpc", "sst2", "qnli"] else evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
        
        if dataset_name in ["mrpc", "sst2", "qnli"]:
            dataset = load_dataset("glue", dataset_name)
        elif dataset_name == "rotten_tomatoes":
            dataset = load_dataset('rotten_tomatoes')
        elif dataset_name == "tweet_eval":
            dataset = load_dataset("tweet_eval", "sentiment")
        elif dataset_name == "yahoo_answers_topics":
            dataset = load_dataset("yahoo_answers_topics").rename_column("topic", "label")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        def select_percentage(dataset, percentage=1):
            return dataset.select(range(int(len(dataset) * percentage)))

        if "train" in dataset:
            dataset["train"] = select_percentage(dataset["train"])
        if "validation" in dataset:
            dataset["validation"] = select_percentage(dataset["validation"])
        if "test" in dataset:
            dataset["test"] = select_percentage(dataset["test"])

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        def preprocess_function(examples):
            if "sentence1" in examples and "sentence2" in examples:
                return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
            elif "sentence" in examples:
                return tokenizer(examples["sentence"], truncation=True)
            elif "text" in examples:
                return tokenizer(examples["text"], truncation=True)
            elif "question" in examples and "sentence" in examples:
                return tokenizer(examples["question"], examples["sentence"], truncation=True)
            elif dataset_name == "yahoo_answers_topics":
                return tokenizer(examples["question_content"], truncation=True)
            elif "content" in examples:
                return tokenizer(examples["content"], truncation=True)
            else:
                raise ValueError(f"Dataset not supported: {dataset_name}")

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_dataset = encoded_dataset["train"]
        eval_dataset = encoded_dataset.get("validation", encoded_dataset.get("test"))
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        for p in p_values:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(set(train_dataset["label"]))).to("cuda")

            def select_top_p_neuron_threshold_by_abs(model, p):
                all_params = torch.cat([p.view(-1) for _, p in model.named_parameters()])
                abs_params = torch.abs(all_params)
                k = int(abs_params.shape[0] * p)
                if args.mode == "smallest":
                    topk_values, _ = torch.topk(abs_params, k, largest=False)
                elif args.mode == "largest":
                    topk_values, _ = torch.topk(abs_params, k)
                return topk_values[-1]
                    
            threshold = select_top_p_neuron_threshold_by_abs(model, p)
            print(f"Dataset: {dataset_name}, p: {p}, Threshold: {threshold}")

            def get_masked_grad_hook(mask):
                def hook_fn(grad):
                    return grad * mask
                return hook_fn

            for param in model.parameters():
                if args.mode == "smallest":
                    mask = (torch.abs(param.data) <= threshold).float()
                elif args.mode == "largest":
                    mask = (torch.abs(param.data) >= threshold).float()
                elif args.mode == "random" and p > 0.05:
                    mask = (torch.rand(param.size()) < 0.05).float().to("cuda")
                else:
                    raise ValueError("Unsupported operation.")
                param.register_hook(get_masked_grad_hook(mask))

            training_args = TrainingArguments(
                output_dir=f"{args.output_path}/{args.model_name}/{dataset_name}_p_{p}",
                num_train_epochs=5,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                evaluation_strategy="epoch",
                save_strategy="no",
                logging_dir=f"{args.output_path}/logs/{args.model_name}/{dataset_name}_p_{p}",
                logging_steps=100,
                load_best_model_at_end=False,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            trainer.train()

            metrics = trainer.evaluate()

            results.append({
                "dataset": dataset_name,
                "p": p,
                "accuracy": metrics.get("eval_accuracy", None),
                "loss": metrics.get("eval_loss", None),
            })
            model.to("cpu")
            del model
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(args.output_path + "/final_results.csv", index=False)
    print(df.pivot(index="p", columns="dataset", values="accuracy"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models with different settings.")
    parser.add_argument("--model_name", type=str, choices=["bert-base-uncased", "roberta-base", "microsoft/deberta-v3-base"], required=True, help="Choose a model.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final results.")
    parser.add_argument("--mode", type=str, required=True, choices=["smallest", "largest", "random"])

    args = parser.parse_args()
    main(args)

