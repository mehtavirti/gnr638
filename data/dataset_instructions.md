# AID Dataset Setup

## Download
Download from the course-provided Google Drive link:
https://drive.google.com/drive/folders/1mX8kaByeedwpp9-4-enuJffGChgmdy9n

## Expected Structure
Place the dataset so it looks like this:

AID/
├── Airport/
│   ├── airport_001.jpg
│   └── ...
├── Beach/
│   ├── beach_001.jpg
│   └── ...
└── ... (30 classes total)

## For Local Use
Set DATA_DIR = '/path/to/AID' in your notebook.

## For Google Colab
Upload to Google Drive and set:
DATA_DIR = '/content/drive/MyDrive/AID'

## Notes
- Do NOT push images to GitHub
- Dataset is image-folder format (torchvision ImageFolder compatible)
- 30 classes, images in .jpg format