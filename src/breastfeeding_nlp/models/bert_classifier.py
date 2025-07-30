"""
Module for fine-tuning BioClinicalBERT for multi-class classification.
This script fine-tunes the BioClinicalBERT model on clinical notes
to classify sentences into feeding-related categories.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from pydantic import BaseModel


class FineTuningConfig(BaseModel):
    """Configuration for fine-tuning BioClinicalBERT."""
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    num_epochs: int = 4
    early_stopping_patience: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_save_path: str = "models/finetuned_bert.pt"
    label_encoder_path: str = "models/label_encoder.pkl"
    random_seed: int = 42


class SentenceDataset(Dataset):
    """Dataset for loading sentences and labels for BERT fine-tuning."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize the dataset with texts and corresponding labels.
        
        Args:
            texts: List of texts to classify
            labels: Corresponding class labels (encoded as integers)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Convert to PyTorch tensors
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTFineTuner:
    """Class for fine-tuning BERT models for classification tasks."""
    
    def __init__(self, config: FineTuningConfig, num_labels: int):
        """
        Initialize the BERT fine-tuner.
        
        Args:
            config: Configuration for fine-tuning
            num_labels: Number of output classes
        """
        self.config = config
        self.num_labels = num_labels
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Move model to the specified device
        self.model = self.model.to(config.device)
        
    def train(self, train_dataloader, val_dataloader):
        """
        Fine-tune the BERT model.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            
        Returns:
            Tuple containing:
            - Fine-tuned model
            - Training loss history
            - Validation loss history
        """
        # Prepare optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Total number of training steps
        total_steps = len(train_dataloader) * self.config.num_epochs
        
        # Set up learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # For early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print(f"Starting fine-tuning on {self.config.device}")
        print(f"Training on {len(train_dataloader.dataset)} samples")
        print(f"Validating on {len(val_dataloader.dataset)} samples")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Training phase
            self.model.train()
            running_loss = 0
            
            # Progress bar for training
            progress_bar = range(len(train_dataloader))
            print("Training...")
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                # Clear previous gradients
                self.model.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                running_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update parameters
                optimizer.step()
                
                # Update scheduler
                scheduler.step()
                
                if (batch_idx + 1) % 20 == 0:
                    print(f"  Batch {batch_idx+1}/{len(train_dataloader)} - Loss: {loss.item():.4f}")
            
            epoch_train_loss = running_loss / len(train_dataloader)
            train_losses.append(epoch_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            print("Validating...")
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                val_loss += outputs.loss.item()
            
            epoch_val_loss = val_loss / len(val_dataloader)
            val_losses.append(epoch_val_loss)
            
            print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
            
            # Early stopping check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                
                # Save the best model
                os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.config.model_save_path)
                print(f"Model saved to {self.config.model_save_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load the best model
        self.model.load_state_dict(torch.load(self.config.model_save_path))
        
        return self.model, train_losses, val_losses
    
    def evaluate(self, test_dataloader, label_encoder):
        """
        Evaluate the fine-tuned model on test data.
        
        Args:
            test_dataloader: DataLoader for test data
            label_encoder: Encoder used to convert class labels to indices
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            labels = batch['labels'].cpu().numpy()
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
        
        # Convert indices back to original class labels
        pred_labels = label_encoder.inverse_transform(all_preds)
        true_labels = label_encoder.inverse_transform(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        f1_macro = f1_score(true_labels, pred_labels, average='macro')
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
        
        # Detailed classification report
        report = classification_report(true_labels, pred_labels, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report,
            'predictions': list(zip(true_labels, pred_labels))
        }
        
        return results


def fine_tune_bert(
    train_data_path: str,
    text_col: str = 'sentence',
    label_col: str = 'sentence_classification',
    train_indices: Optional[np.ndarray] = None,
    val_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
    val_split: float = 0.2,
    test_split: float = 0.1,
    config: Optional[FineTuningConfig] = None
) -> Tuple[BERTFineTuner, LabelEncoder, Dict]:
    """
    Fine-tune BioClinicalBERT for text classification.
    
    Args:
        train_data_path: Path to the training data CSV file
        text_col: Column name containing the text
        label_col: Column name containing the labels
        train_indices: Specific indices to use for training (optional)
        val_indices: Specific indices to use for validation (optional)
        test_indices: Specific indices to use for testing (optional)
        val_split: Proportion of data to use for validation (if indices not provided)
        test_split: Proportion of data to use for testing (if indices not provided)
        config: Configuration for fine-tuning (optional)
    
    Returns:
        Tuple containing:
        - Fine-tuner object with trained model
        - Label encoder
        - Evaluation results
    """
    # Load data
    print(f"Loading data from {train_data_path}")
    df = pd.read_csv(train_data_path)
    
    # Encode class labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df[label_col])
    
    # Initialize config if not provided
    if config is None:
        config = FineTuningConfig()
    
    # Print information about the classes
    num_labels = len(label_encoder.classes_)
    print(f"Number of classes: {num_labels}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Initialize BERT fine-tuner
    fine_tuner = BERTFineTuner(config, num_labels)
    
    # Split data into train, validation, and test sets if indices not provided
    if train_indices is None or val_indices is None or test_indices is None:
        print("Using automatic split based on provided proportions")
        n_samples = len(df)
        indices = np.random.permutation(n_samples)
        
        test_size = int(test_split * n_samples)
        val_size = int(val_split * n_samples)
        train_size = n_samples - val_size - test_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
    else:
        print("Using provided indices for train/validation/test split")
    
    # Create datasets
    train_texts = df[text_col].iloc[train_indices].values
    train_labels = encoded_labels[train_indices]
    
    val_texts = df[text_col].iloc[val_indices].values
    val_labels = encoded_labels[val_indices]
    
    test_texts = df[text_col].iloc[test_indices].values
    test_labels = encoded_labels[test_indices]
    
    print(f"Train set: {len(train_texts)} samples")
    print(f"Validation set: {len(val_texts)} samples")
    print(f"Test set: {len(test_texts)} samples")
    
    # Create datasets
    train_dataset = SentenceDataset(
        train_texts, 
        train_labels, 
        fine_tuner.tokenizer, 
        max_length=config.max_length
    )
    val_dataset = SentenceDataset(
        val_texts, 
        val_labels, 
        fine_tuner.tokenizer, 
        max_length=config.max_length
    )
    test_dataset = SentenceDataset(
        test_texts, 
        test_labels, 
        fine_tuner.tokenizer, 
        max_length=config.max_length
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size
    )
    
    # Fine-tune the model
    model, train_losses, val_losses = fine_tuner.train(train_dataloader, val_dataloader)
    
    # Evaluate the model
    results = fine_tuner.evaluate(test_dataloader, label_encoder)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score (macro): {results['f1_macro']:.4f}")
    print(f"F1 Score (weighted): {results['f1_weighted']:.4f}")
    
    # Save label encoder
    import joblib
    os.makedirs(os.path.dirname(config.label_encoder_path), exist_ok=True)
    joblib.dump(label_encoder, config.label_encoder_path)
    print(f"Label encoder saved to {config.label_encoder_path}")
    
    # Save the model
    fine_tuner.model.save_pretrained(os.path.dirname(config.model_save_path))
    
    return fine_tuner, label_encoder, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fine-tune BioClinicalBERT for text classification"
    )
    parser.add_argument(
        "--train_data", 
        type=str, 
        default="examples/train_df.csv",
        help="Path to the training data CSV file"
    )
    parser.add_argument(
        "--text_col", 
        type=str, 
        default="sentence",
        help="Column name containing the text"
    )
    parser.add_argument(
        "--label_col", 
        type=str, 
        default="sentence_classification",
        help="Column name containing the labels"
    )
    parser.add_argument(
        "--indices_file", 
        type=str, 
        help="Path to a numpy file (.npz) containing train_indices, val_indices, and test_indices"
    )
    parser.add_argument(
        "--val_split", 
        type=float, 
        default=0.2,
        help="Proportion of data to use for validation (if indices not provided)"
    )
    parser.add_argument(
        "--test_split", 
        type=float, 
        default=0.1,
        help="Proportion of data to use for testing (if indices not provided)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=4,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=2e-5,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="models/finetuned_bert",
        help="Path to save the trained model"
    )
    
    args = parser.parse_args()
    
    # Create config with command line arguments
    config = FineTuningConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        model_save_path=args.model_path
    )
    
    # Load indices if provided
    train_indices = None
    val_indices = None
    test_indices = None
    
    if args.indices_file:
        try:
            indices_data = np.load(args.indices_file)
            train_indices = indices_data['train_indices']
            val_indices = indices_data['val_indices']
            test_indices = indices_data['test_indices']
            print(f"Loaded indices from {args.indices_file}")
        except Exception as e:
            print(f"Error loading indices from {args.indices_file}: {e}")
            print("Falling back to automatic split")
    
    # Run fine-tuning
    fine_tune_bert(
        args.train_data,
        args.text_col,
        args.label_col,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        val_split=args.val_split,
        test_split=args.test_split,
        config=config
    ) 