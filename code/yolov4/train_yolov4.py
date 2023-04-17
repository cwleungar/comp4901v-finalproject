import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.tensorboard as tb
import torchvision.models.detection as detection
from ..Dataset import ObjectDetectionDataset   # Import your custom dataset module
from . import  yolov4             # Import your YOLOv4 module
import torchvision.transforms as T
import argparse
from os import path
import time
current_GMT = time.time()

def train(args):
    print("inside train")
    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameters
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    num_classes = args.num_classes
    data_path=args.data_path
    label_path=args.label_path
    # Create dataset and data loaders
    train_dataset = ObjectDetectionDataset(data_path,label_path,split='train', val_split=0.1, transform=T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomCrop(size=(224, 224)),
        T.ToTensor()
    ]))
    val_dataset = ObjectDetectionDataset(data_path,label_path,split='val', val_split=0.1, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create YOLOv4 model and optimizer
    model = yolov4.YOLOv4(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    # Create loss function
    criterion = nn.MSELoss()
    logger = tb.SummaryWriter(path.join(args.log_dir, f'yolov4-lr-{args.learning_rate}-wd--{args.weight_decay}-{current_GMT}'), flush_secs=1)

    # Create TensorBoard writer

    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        # Loop over training batches
        for i, (images, targets) in enumerate(train_loader):
            # Move inputs and targets to device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training loss to TensorBoard
            logger.add_scalar('train/Training Loss', loss.item(), epoch * len(train_loader) + i)
            # Compute mAP using torchvision's detection API
            detection_model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
            detection_model.eval()
            detection_model.to(device)
            gt_boxes = [b.to(device) for b in gt_boxes]
            pred_boxes = [b.to(device) for b in pred_boxes]
            mean_average_precision = detection.coco_evaluation.evaluate(
                gt_boxes, pred_boxes, iou_threshold=0.5
            )['map']

            # Log mAP to TensorBoard
            logger.add_scalar('train/mAP', mean_average_precision, epoch)
        # Set model to evaluation mode
        model.eval()

        # Evaluate on validation set
        with torch.no_grad():
            total_loss = 0.0
            gt_boxes = []
            pred_boxes = []
            for images, targets in val_loader:
                # Move inputs and targets to device
                images = images.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * len(images)

                # Compute predicted boxes and append to list
                pred_boxes += [model.decode_output(output) for output in outputs]

                # Append ground truth boxes to list
                for target in targets:
                    gt_boxes.append({
                        'boxes': target[:, :4],
                        'labels': target[:, 4]
                    })

            # Compute average validation loss
            avg_val_loss = total_loss / len(val_dataset)

            # Log validation loss to TensorBoard
            logger.add_scalar('val/Validation Loss', avg_val_loss, epoch)

            # Compute mAP using torchvision's detection API
            detection_model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
            detection_model.eval()
            detection_model.to(device)
            gt_boxes = [b.to(device) for b in gt_boxes]
            pred_boxes = [b.to(device) for b in pred_boxes]
            mean_average_precision = detection.coco_evaluation.evaluate(
                gt_boxes, pred_boxes, iou_threshold=0.5
            )['map']

            # Log mAP to TensorBoard
            logger.add_scalar('val/mAP', mean_average_precision, epoch)

        # Save model checkpoint
        torch.save(model.state_dict(), f'model_checkpoint_{epoch}.pt')

    # Close TensorBoard writer
    logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv4 Training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes')
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset')
    parser.add_argument('--label_path', type=str, default='label', help='Path to label')
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to logs')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    args = parser.parse_args()
    print("train start")
    train(args)
