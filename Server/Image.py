<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Description</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>Image Description</h1>
        <div class="input-container">
            <input type="file" id="imageInput" accept="image/*">
            <button id="submitBtn">Submit</button>
        </div>
        <div class="output-container">
            <img id="imagePreview" src="" alt="Image Preview">
            <p id="description"></p>
        </div>
    </div>
    <script>
        const imageInput = document.getElementById('imageInput');
const submitBtn = document.getElementById('submitBtn');
const imagePreview = document.getElementById('imagePreview');
const description = document.getElementById('description');

imageInput.addEventListener('change', () => {
    const file = imageInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = () => {
            imagePreview.src = reader.result;
        };
        reader.readAsDataURL(file);
    }
});

submitBtn.addEventListener('click', async () => {
    const file = imageInput.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('http://localhost:5000/image_description', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            description.textContent = data.description;
        } catch (error) {
            console.error('Error:', error);
        }
    }
});
    </script>
</body>
</html>
