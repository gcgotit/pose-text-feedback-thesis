from pathlib import Path
import argparse

# ==============================================================================
# EC3D Annotation Generator
# Supports both WITH_UNKNOWN (12 classes) and NO_UNKNOWN (11 classes) versions
# ==============================================================================

# Mapping EC3D WITH Unknown: (exercise, instruction_id) -> global class_id
GLOBAL_LABEL_MAP_WITH_UNKNOWN = {
    ('SQUAT', 1): 0,   ('SQUAT', 2): 1, ('SQUAT', 3): 2, ('SQUAT', 4): 3,
    ('SQUAT', 5): 4,   ('SQUAT', 10): 5,
    ('Lunges', 1): 6,  ('Lunges', 4): 7, ('Lunges', 6): 8,
    ('Plank', 1): 9,   ('Plank', 7): 10, ('Plank', 8): 11,
}

# Mapping EC3D NO Unknown: (exercise, instruction_id) -> global class_id
# Label 5 removed, labels 6-11 shifted to 5-10
GLOBAL_LABEL_MAP_NO_UNKNOWN = {
    ('SQUAT', 1): 0,   ('SQUAT', 2): 1, ('SQUAT', 3): 2, ('SQUAT', 4): 3,
    ('SQUAT', 5): 4,
    ('Lunges', 1): 5,  ('Lunges', 4): 6, ('Lunges', 6): 7,
    ('Plank', 1): 8,   ('Plank', 7): 9, ('Plank', 8): 10,
}

# Descrizioni sintetiche stile FLAG3D - WITH Unknown (12 classes)
DESCRIPTIONS_WITH_UNKNOWN = {
    0: ("Squat (Correct)", "Stand tall, bend your knees and lower into a squat keeping the back straight.",
        "Inhale while lowering, exhale while standing.",
        "Keep knees in line with toes, chest up.",
        "mistake: leaning forward — solution: engage core and lift chest.",
        "Keep heels on the ground throughout the movement."),

    1: ("Squat (Feet too wide)", "Perform a squat with feet excessively spread apart.",
        "Inhale while lowering, exhale while rising.",
        "Adjust stance to shoulder-width.",
        "mistake: feet too wide — solution: bring feet closer.",
        "Excessive width can reduce stability and depth."),

    2: ("Squat (Knees inward)", "Perform a squat where the knees cave inward during descent.",
        "Inhale on the way down, exhale on the way up.",
        "Drive knees outward.",
        "mistake: knees collapsing — solution: cue to push knees out.",
        "Often due to weak glutes or improper stance."),

    3: ("Squat (Not low enough)", "Perform a shallow squat with limited depth.",
        "Breathe normally.",
        "Aim for thighs parallel to the ground.",
        "mistake: stopping too high — solution: increase range gradually.",
        "Flexibility or fear can limit depth."),

    4: ("Squat (Front bended)", "Squat with torso leaning excessively forward.",
        "Exhale on ascent.",
        "Maintain upright chest.",
        "mistake: bending forward — solution: strengthen core, adjust bar position.",
        "May indicate limited ankle mobility."),

    5: ("Squat (Unknown error)", "Squat form is invalid for unknown reasons.",
        "Breathe comfortably.",
        "Ensure neutral spine and controlled motion.",
        "mistake: unclear — solution: seek coach review.",
        "Use video feedback for form analysis."),

    6: ("Lunges (Correct)", "Step forward and lower until both knees are at 90 degrees.",
        "Inhale as you descend, exhale returning.",
        "Keep torso upright, front knee over ankle.",
        "mistake: unstable balance — solution: shorten step.",
        "Use arms for stability if needed."),

    7: ("Lunges (Not low enough)", "Perform a lunge with reduced knee bend.",
        "Normal breathing.",
        "Strive for both knees at 90 degrees.",
        "mistake: partial depth — solution: practice mobility drills.",
        "Focus on lowering hips directly down."),

    8: ("Lunges (Knees pass toes)", "Front knee moves too far past toes.",
        "Inhale down, exhale up.",
        "Maintain vertical shin.",
        "mistake: knee forward — solution: shorten stride.",
        "Can stress the knee joint unnecessarily."),

    9: ("Plank (Correct)", "Hold a straight line from head to heels, elbows under shoulders.",
        "Breathe slowly and evenly.",
        "Engage core and glutes.",
        "mistake: holding breath — solution: steady breathing.",
        "Look down to maintain neutral neck."),

    10: ("Plank (Banana back)", "Lower back sags forming an arch.",
         "Exhale to engage core.",
         "Tuck pelvis under slightly.",
         "mistake: sagging — solution: tighten core.",
         "Modify by dropping knees if needed."),

    11: ("Plank (Rolled back)", "Hips too high breaking the plank line.",
         "Inhale through the nose, exhale through the mouth.",
         "Lower hips to align with shoulders.",
         "mistake: raised hips — solution: keep back flat.",
         "Watch yourself in a mirror to correct.")
}

# Descrizioni sintetiche stile FLAG3D - NO Unknown (11 classes)
# Same content, but remapped class IDs
DESCRIPTIONS_NO_UNKNOWN = {
    0: ("Squat (Correct)", "Stand tall, bend your knees and lower into a squat keeping the back straight.",
        "Inhale while lowering, exhale while standing.",
        "Keep knees in line with toes, chest up.",
        "mistake: leaning forward — solution: engage core and lift chest.",
        "Keep heels on the ground throughout the movement."),

    1: ("Squat (Feet too wide)", "Perform a squat with feet excessively spread apart.",
        "Inhale while lowering, exhale while rising.",
        "Adjust stance to shoulder-width.",
        "mistake: feet too wide — solution: bring feet closer.",
        "Excessive width can reduce stability and depth."),

    2: ("Squat (Knees inward)", "Perform a squat where the knees cave inward during descent.",
        "Inhale on the way down, exhale on the way up.",
        "Drive knees outward.",
        "mistake: knees collapsing — solution: cue to push knees out.",
        "Often due to weak glutes or improper stance."),

    3: ("Squat (Not low enough)", "Perform a shallow squat with limited depth.",
        "Breathe normally.",
        "Aim for thighs parallel to the ground.",
        "mistake: stopping too high — solution: increase range gradually.",
        "Flexibility or fear can limit depth."),

    4: ("Squat (Front bended)", "Squat with torso leaning excessively forward.",
        "Exhale on ascent.",
        "Maintain upright chest.",
        "mistake: bending forward — solution: strengthen core, adjust bar position.",
        "May indicate limited ankle mobility."),

    # Class 5 (Unknown) is REMOVED - Lunges start at 5

    5: ("Lunges (Correct)", "Step forward and lower until both knees are at 90 degrees.",
        "Inhale as you descend, exhale returning.",
        "Keep torso upright, front knee over ankle.",
        "mistake: unstable balance — solution: shorten step.",
        "Use arms for stability if needed."),

    6: ("Lunges (Not low enough)", "Perform a lunge with reduced knee bend.",
        "Normal breathing.",
        "Strive for both knees at 90 degrees.",
        "mistake: partial depth — solution: practice mobility drills.",
        "Focus on lowering hips directly down."),

    7: ("Lunges (Knees pass toes)", "Front knee moves too far past toes.",
        "Inhale down, exhale up.",
        "Maintain vertical shin.",
        "mistake: knee forward — solution: shorten stride.",
        "Can stress the knee joint unnecessarily."),

    8: ("Plank (Correct)", "Hold a straight line from head to heels, elbows under shoulders.",
        "Breathe slowly and evenly.",
        "Engage core and glutes.",
        "mistake: holding breath — solution: steady breathing.",
        "Look down to maintain neutral neck."),

    9: ("Plank (Banana back)", "Lower back sags forming an arch.",
         "Exhale to engage core.",
         "Tuck pelvis under slightly.",
         "mistake: sagging — solution: tighten core.",
         "Modify by dropping knees if needed."),

    10: ("Plank (Rolled back)", "Hips too high breaking the plank line.",
         "Inhale through the nose, exhale through the mouth.",
         "Lower hips to align with shoulders.",
         "mistake: raised hips — solution: keep back flat.",
         "Watch yourself in a mirror to correct.")
}

# For backwards compatibility
GLOBAL_LABEL_MAP = GLOBAL_LABEL_MAP_WITH_UNKNOWN
DESCRIPTIONS = DESCRIPTIONS_WITH_UNKNOWN

def generate_annotations(out_dir: Path, descriptions: dict, include_unknown: bool = True):
    """
    Generate FLAG3D-style annotation files.
    
    Args:
        out_dir: Output directory
        descriptions: Dictionary of class_id -> (name, desc, breath, keypts, errors, notes)
        include_unknown: If True, generating with_unknown version
    """
    out_dir.mkdir(exist_ok=True)
    
    # Genera i file AxxxI00n.txt stile FLAG3D
    for class_id, (name, desc, breath, keypts, errors, notes) in descriptions.items():
        prefix = f"A{class_id:03d}"
        (out_dir / f"{prefix}I001.txt").write_text(name)
        (out_dir / f"{prefix}I002.txt").write_text(desc)
        (out_dir / f"{prefix}I003.txt").write_text(breath)
        (out_dir / f"{prefix}I004.txt").write_text(keypts)
        (out_dir / f"{prefix}I005.txt").write_text(errors)
        (out_dir / f"{prefix}I006.txt").write_text(notes)
    
    # File con template generico
    template_filename = "generic_templates.txt"
    with open(out_dir / template_filename, "w") as f:
        for class_id in sorted(descriptions.keys()):
            name = descriptions[class_id][0]
            f.write(f"The subject is performing {name}\n")
    
    version = "with Unknown (12 classes)" if include_unknown else "NO Unknown (11 classes)"
    print(f"✅ Annotazioni FLAG3D-like salvate in {out_dir.resolve()} [{version}]")


def main():
    parser = argparse.ArgumentParser(description="Generate EC3D annotations in FLAG3D style")
    parser.add_argument(
        "--no-unknown", 
        action="store_true",
        help="Generate annotations without Unknown class (11 classes, paper-aligned)"
    )
    parser.add_argument(
        "--both",
        action="store_true", 
        help="Generate both WITH and WITHOUT Unknown versions"
    )
    args = parser.parse_args()
    
    # Base output directory (relative to script location or data dir)
    base_dir = Path(__file__).parent.parent.parent / "data"
    
    if args.both:
        # Generate both versions
        generate_annotations(
            base_dir / "annotations_ec3d", 
            DESCRIPTIONS_WITH_UNKNOWN, 
            include_unknown=True
        )
        generate_annotations(
            base_dir / "annotations_ec3d_no_unknown", 
            DESCRIPTIONS_NO_UNKNOWN, 
            include_unknown=False
        )
    elif args.no_unknown:
        # Generate only no_unknown version
        generate_annotations(
            base_dir / "annotations_ec3d_no_unknown", 
            DESCRIPTIONS_NO_UNKNOWN, 
            include_unknown=False
        )
    else:
        # Default: generate with_unknown version (backwards compatible)
        generate_annotations(
            base_dir / "annotations_ec3d", 
            DESCRIPTIONS_WITH_UNKNOWN, 
            include_unknown=True
        )


if __name__ == "__main__":
    main()
