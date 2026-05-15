import pandas as pd
from database.db import init_db, SessionLocal, CXRCase

CHEXPERT_CSV = "data/CheXpert-v1.0-small/train.csv"
SAMPLE_SIZE  = 500

LABEL_COLUMNS = [
    "pneumonia", "cardiomegaly", "pleural_effusion",
    "pneumothorax", "atelectasis", "lung_mass"
]


def seed_cases():
    init_db()
    db = SessionLocal()

    # Avoid re-seeding if already populated
    if db.query(CXRCase).count() > 0:
        print("Already seeded, skipping.")
        db.close()
        return

    # TODO: read CSV with pd.read_csv
    df = pd.read_csv(CHEXPERT_CSV)
    # TODO: sample SAMPLE_SIZE rows with .sample(n=SAMPLE_SIZE, random_state=42)
    sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    # TODO: loop through rows and insert CXRCase objects
    cases = []
    for idx, row in sample.iterrows():
        case = CXRCase(
            image_id = str(row["Path"]),
            patient_id = str(row["Path"]).split("/")[2],
            image_path = str(row["Path"]),
            embedding_index = idx,
            pneumonia = int(max(0, row.get("Pneumonia", 0) or 0)),
            cardiomegaly = int(max(0, row.get("Cardiomegaly", 0) or 0)),
            pleural_effusion = int(max(0, row.get("Pleural Effusion", 0) or 0)),
            pneumothorax = int(max(0, row.get("Pneumothorax", 0) or 0)),
            atelectasis = int(max(0, row.get("Atelectasis", 0) or 0)),
            lung_mass = int(max(0, row.get("Lung Lesion", 0) or 0)),
        )
        cases.append(case)
    # TODO: db.add + db.commit at the end
    db.add_all(cases)
    db.commit()
    # hint: use iterrows() to loop — for idx, row in df.iterrows()

    db.close()
    print(f"Seeded {SAMPLE_SIZE} cases.")


if __name__ == "__main__":
    seed_cases()