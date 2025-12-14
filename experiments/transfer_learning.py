from transformers import AutoImageProcessor, AutoModelForImageClassification

# based on EfficientNet and trained on bird species dataset
processor_effnet = AutoImageProcessor.from_pretrained("chriamue/bird-species-classifier")
model_effnet = AutoModelForImageClassification.from_pretrained("chriamue/bird-species-classifier", num_labels=200, ignore_mismatched_sizes=True)

# based on ViT and trained on bird species dataset
processor_vit = AutoImageProcessor.from_pretrained("Hemg/Birds-Species-classification")
model_vit = AutoModelForImageClassification.from_pretrained("Hemg/Birds-Species-classification", num_labels=200, ignore_mismatched_sizes=True)

# based on Swin transformers and trained on ImageNet
processor_swin = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
model_swin = AutoModelForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224", num_labels=200, ignore_mismatched_sizes=True)