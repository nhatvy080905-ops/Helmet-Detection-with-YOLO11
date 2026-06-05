import os

# class cũ của dataset cần gộp
old_names = ['helmet', 'nohelmet', 'numberplate', 'rider']

# class mới sau khi gộp
new_names = ['helmet', 'no_helmet']

# ánh xạ tên class cũ sang tên class mới
name_map = {
    'helmet': 'helmet',
    'nohelmet': 'no_helmet',
    'numberplate': None,
    'rider': None
}

label_dir = r"D:\Projetc_Helmet\Rider-1\train\labels"

for file_name in os.listdir(label_dir):
    if not file_name.endswith(".txt"):
        continue

    label_path = os.path.join(label_dir, file_name)

    new_lines = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        old_class_id = int(parts[0])
        old_class_name = old_names[old_class_id]

        new_class_name = name_map.get(old_class_name)

        # Nếu class này không cần dùng thì bỏ qua
        if new_class_name is None:
            continue

        new_class_id = new_names.index(new_class_name)

        # thay class_id cũ bằng class_id mới
        parts[0] = str(new_class_id)

        new_lines.append(" ".join(parts))

    with open(label_path, "w") as f:
        f.write("\n".join(new_lines))