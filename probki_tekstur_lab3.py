import os
import cv2


def extract_texture_samples(input_dir, output_dir, sample_size):
    # Sprawdź, czy katalog wynikowy istnieje, jeśli nie, utwórz go
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lista kategorii tekstur
    categories = os.listdir(input_dir)

    for category in categories:
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue

        output_category_dir = os.path.join(output_dir, category)
        if not os.path.exists(output_category_dir):
            os.makedirs(output_category_dir)

        # Lista plików w bieżącej kategorii
        files = os.listdir(category_path)

        for file_name in files:
            file_path = os.path.join(category_path, file_name)

            # Wczytaj obraz
            image = cv2.imread(file_path)

            # Wycinanie fragmentów tekstury
            height, width, _ = image.shape
            for y in range(0, height - sample_size + 1, sample_size):
                for x in range(0, width - sample_size + 1, sample_size):
                    sample = image[y:y + sample_size, x:x + sample_size]

                    # Nazwa pliku dla wyciętego fragmentu
                    sample_name = os.path.splitext(file_name)[0] + f'_sample_{y}_{x}.jpg'
                    output_sample_path = os.path.join(output_category_dir, sample_name)

                    # Zapisz wycięty fragment
                    cv2.imwrite(output_sample_path, sample)


# Przykładowe użycie
input_directory = 'tekstury'
output_directory = 'probki'
sample_size = 128

extract_texture_samples(input_directory, output_directory, sample_size)
