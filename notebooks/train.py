import os

import pandas as pd
import torch
from fire import Fire
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import clip
import wandb


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

        return image, caption


def main(
    image_dir="/home/ghoti/llm/transformer/clip_learn/",
    train_csv="~/llm/transformer/clip_learn/data/flickr8k_train.csv",
    val_csv="~/llm/transformer/clip_learn/data/flickr8k_val.csv",
    save_dir="/home/ghoti/CLIP_test/models",
    model_name="ViT-B/32",
    batch_size=8,
    epochs=5,
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
    print("Device:", device)

    model, preprocess = clip.load(model_name, device=device)

    train_dataset = FlickrDataset(train_csv, image_dir, transform=preprocess)
    val_dataset = FlickrDataset(val_csv, image_dir, transform=preprocess)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for i, (images, texts) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device)

            I_f = model.encode_image(images)
            T_f = model.encode_text(texts)

            I_e = I_f / I_f.norm(dim=-1, keepdim=True)
            T_e = T_f / T_f.norm(dim=-1, keepdim=True)

            logits = (I_e @ T_e.T) * model.logit_scale.exp()

            labels = torch.arange(len(images)).to(device)
            loss_i = criterion(logits, labels)
            loss_t = criterion(logits.T, labels)
            loss = (loss_i + loss_t) / 2

            loss.backward()
            optimizer.step()

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
