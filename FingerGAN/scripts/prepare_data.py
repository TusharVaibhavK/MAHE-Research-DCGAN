import os
import csv

# Paths
SOCO_PATH = "data/SOCOFing"
FAMILY_PATH = "data/FAMILY_FINGERPRINT_DATASET"
OUT_CSV = "data/combined_metadata.csv"


def parse_soco_filename(fname):
    """
    Example: 001__M_Left_index_finger.BMP
    Extract subject_id, gender, finger
    """
    parts = fname.split("__")
    if len(parts) < 2:
        return None, None, None
    subject_id = parts[0]
    rest = parts[1]
    bits = rest.split("_")
    if len(bits) >= 2:
        gender = bits[0]
        finger = "_".join(bits[1:])
    else:
        gender, finger = None, None
    return subject_id, gender, finger


def collect_soco(writer):
    for root, _, files in os.walk(SOCO_PATH):
        for f in files:
            if f.lower().endswith(".bmp"):
                subject_id, gender, finger = parse_soco_filename(f)
                filepath = os.path.join(root, f).replace("\\", "/")
                writer.writerow(
                    [filepath, "Africa", "SOCOFing", subject_id, "", "", finger])


def collect_family(writer):
    for family in os.listdir(FAMILY_PATH):
        fam_path = os.path.join(FAMILY_PATH, family)
        if not os.path.isdir(fam_path):
            continue
        for relation in os.listdir(fam_path):
            rel_path = os.path.join(fam_path, relation)
            if not os.path.isdir(rel_path):
                continue
            for f in os.listdir(rel_path):
                if f.lower().endswith(".bmp"):
                    filepath = os.path.join(rel_path, f).replace("\\", "/")
                    subject_id = f"{family}_{relation}"
                    writer.writerow(
                        [filepath, "SouthAsia", "Family", subject_id, family, relation, ""])


def main():
    os.makedirs("data", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "region", "dataset",
                        "subject_id", "family_id", "relation", "finger"])
        collect_soco(writer)
        collect_family(writer)
    print(f"Metadata written to {OUT_CSV}")


if __name__ == "__main__":
    main()
