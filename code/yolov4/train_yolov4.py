import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.models.detection as detection
from .. import Dataset   # Import your custom dataset module
from yolov4 import YOLOv4             # Import your YOLOv4 module
import torchvision.transforms as T
# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
batch_size = 16
num_epochs = 100
learning_rate = 0.001
num_classes = 3

# Create dataset and data loaders
train_dataset = Dataset.ObjectDetectionDataset('path/to/train/dataset',split='train', val_split=0.1, transform=T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomCrop(size=(224, 224)),
    T.ToTensor()
]))
val_dataset = Dataset.ObjectDetectionDataset('path/to/val/dataset',split='val', val_split=0.1, transform=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create YOLOv4 model and optimizer
model = YOLOv4(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create loss function
criterion = nn.MSELoss()

# Create TensorBoard writer
writer = SummaryWriter()

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
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)
    
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
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        
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
        writer.add_scalar('mAP', mean_average_precision, epoch)
        
    # Save model checkpoint
    torch.save(model.state_dict(), f'model_checkpoint_{epoch}.pt')
    
# Close TensorBoard writer
writer.close()