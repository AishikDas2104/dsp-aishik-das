install:
	pip install -r requirements.txt

train:
	python main.py train --data data/train.csv

predict:
	python main.py predict --input data/test.csv --output predictions.csv

test:
	pytest
