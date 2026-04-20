### in sample-train-1 we immediately use a sabotaged training-process-only-MODEL-AGNOSTICISM

import json
import time
import os
import subprocess


## Simulate training
def train_dummy_model():
    print("Starting ML Training Run...")
    logs = {"epochs":[]}
    
    # massive overfitting
    train_loss = 2.0
    val_loss = 2.0
    
    for epoch in range(1, 21):
        
        train_loss = train_loss * -1.0 
        # Val loss goes down, then spikes after epoch 10 (Overfitting)
        val_loss = val_loss * 0.88 if epoch < 10 else val_loss * 1.15 
        
        logs["epochs"].append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "grad_norm": round(0.5 / epoch, 4) # Gradients vanishing
        })
        print(f"Epoch {epoch}/20 - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        time.sleep(0.1) # Simulate training time
        
    with open("model/logs/training_logs.json", "w") as f:
        json.dump(logs, f, indent=4)
    with open("model/models_save/model.pt", "w") as f:
        f.write("dummy_model_weights")
        
    print("\n Training Complete. Saved 'model.pt' and 'training_logs.json'.")
    
    print("Launching <Parallel> ...")
    subprocess.Popen(["streamlit", "run", "main_stream.py"])

if __name__ == "__main__":
    train_dummy_model()