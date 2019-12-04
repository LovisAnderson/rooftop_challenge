# Rooftop Detection Challenge

## Installation
- Create virtual environment (python <=3.6) `virtualenv venv`
- Activate environment (bash) `source venv/bin/activate`
- Install requirements `pip install -r requirements.txt`

## Start training
- `python segmentation.py`

## I used
- Pretrained model from https://github.com/qubvel/segmentation_models
- Additional data from  https://www.airs-dataset.com/ (I did not have the time to take full advantage of it)
- https://dataturks.com/ for annotation of validation images (I did not train on them, just for easier model evaluation)

## Next steps
- make more use of large christchurch dataset (prepare/augment data better)
- Try different models/losses (dice loss seems better from intuition since it aligns with IoU score which is very informative in my opinion)