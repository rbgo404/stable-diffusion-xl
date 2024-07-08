INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Penguins having dinner"]
    },
    "negative_prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["low quality, low resolution, greyscale, multiple fingers, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, deformed toes standing still, posing"]
    }
}
