import os

import pandas as pd
import torch
from fire import Fire
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import clip
import wandb

torch.autograd.set_detect_anomaly(True)


class FlickrDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filepath"])
        caption = row["title"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        if self.transform:
            image = self.transform(image)
        caption = clip.tokenize([caption])[0]

        if image is None or caption is None:
            raise ValueError(f"Invalid data at index {idx}: {row}")

        return image, caption


def main(
    image_dir="/home/mdxuser/data/flickr8k/images",
    train_csv="/home/mdxuser/data/flickr8k_train.csv",
    val_csv="/home/mdxuser/data/flickr8k_val.csv",
    save_dir="/home/mdxuser/CLIP_test/models",
    model_name="ViT-B/32",
    batch_size=128,
    epochs=10,
    learning_rate=1e-5,
):
    project_name = "CLIP-test-run"
    wandb.login(key="c85b817c62f441243d232b381088358e72fa2b19")
    wandb.init(
        project=project_name,
        config={
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model, preprocess = clip.load(model_name, device=device, jit=False)
    model = model.float()
    model.logit_scale.data = torch.clamp(model.logit_scale.data, max=100)
    train_dataset = FlickrDataset(train_csv, image_dir, transform=preprocess)
    val_dataset = FlickrDataset(val_csv, image_dir, transform=preprocess)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (images, texts) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device, dtype=torch.float32)
            texts = texts.to(device, dtype=torch.long)
            I_f = model.encode_image(images)
            T_f = model.encode_text(texts)
            I_e = I_f / I_f.norm(dim=-1, keepdim=True)
            T_e = T_f / T_f.norm(dim=-1, keepdim=True)
            logits = (I_e @ T_e.T) * model.logit_scale.exp()

            labels = torch.arange(len(images)).to(device)
            loss_i = criterion(logits, labels)
            loss_t = criterion(logits.T, labels)
            loss = (loss_i + loss_t) / 2
            if torch.isnan(loss):
                print(f"NaN detected at Epoch [{epoch + 1}], Step [{i}]")
                return
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, texts in val_loader:
                images = images.to(device)
                texts = texts.to(device)
                I_f = model.encode_image(images)

                I_e = I_f / I_f.norm(dim=-1, keepdim=True)
                T_e = T_f / T_f.norm(dim=-1, keepdim=True)

                logits = (I_e @ T_e.T) * model.logit_scale.exp()

                labels = torch.arange(len(images)).to(device)
                loss_i = criterion(logits, labels)
                loss_t = criterion(logits.T, labels)
                val_loss += (loss_i + loss_t) / 2

            val_loss /= len(val_loader)
            print(f"Validation Loss after Epoch [{epoch + 1}/{epochs}]: {val_loss:.4f}")

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

    os.makedirs(os.path.join(save_dir, wandb.run.name), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
        },
        os.path.join(save_dir, wandb.run.name, "model.pth"),
    )
    wandb.finish()


if __name__ == "__main__":
    Fire(main)
