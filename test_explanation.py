# test_explanation.py
from modules.explanation import generate_report

TEST_IMAGE = r"D:\Projek\MedStep\data\CheXpert-v1.0-small\train\patient21923\study2\view1_frontal.jpg"

# Simulate detection results
detections = {
    "pneumonia":        0.92,
    "cardiomegaly":     0.12,
    "pleural_effusion": 0.08,
    "pneumothorax":     0.03,
    "atelectasis":      0.15,
    "lung_mass":        0.06
}

print("Generating report...")
report = generate_report(TEST_IMAGE, detections)

print("\n=== FINDINGS ===")
print(report["findings"])
print("\n=== IMPRESSION ===")
print(report["impression"])
print("\n=== DETECTIONS ===")
print(report["detections_text"])