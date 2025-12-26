from pathlib import Path

# Mapping EC3D: (exercise, instruction_id) -> global class_id
GLOBAL_LABEL_MAP = {
    ('SQUAT', 1): 0,   ('SQUAT', 2): 1, ('SQUAT', 3): 2, ('SQUAT', 4): 3,
    ('SQUAT', 5): 4,   ('SQUAT', 10): 5,
    ('Lunges', 1): 6,  ('Lunges', 4): 7, ('Lunges', 6): 8,
    ('Plank', 1): 9,   ('Plank', 7): 10, ('Plank', 8): 11,
}

# Descrizioni sintetiche stile FLAG3D
DESCRIPTIONS = {
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

# Output directory
out_dir = Path("annotations_ec3d")
out_dir.mkdir(exist_ok=True)

# Genera i file AxxxI00n.txt stile FLAG3D
for class_id, (name, desc, breath, keypts, errors, notes) in DESCRIPTIONS.items():
    prefix = f"A{class_id:03d}"
    (out_dir / f"{prefix}I001.txt").write_text(name)
    (out_dir / f"{prefix}I002.txt").write_text(desc)
    (out_dir / f"{prefix}I003.txt").write_text(breath)
    (out_dir / f"{prefix}I004.txt").write_text(keypts)
    (out_dir / f"{prefix}I005.txt").write_text(errors)
    (out_dir / f"{prefix}I006.txt").write_text(notes)

# File con template generico
with open(out_dir / "generic_templates.txt", "w") as f:
    for class_id, (name, *_rest) in DESCRIPTIONS.items():
        f.write(f"The subject is performing {name}\n")

print(f"✅ Annotazioni FLAG3D-like e template generici salvati in {out_dir.resolve()}")
