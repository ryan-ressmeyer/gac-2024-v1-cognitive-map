# GAC 2024 V1 Cognitive Map

Code for the 2024 Generative Adversarial Collaboration - Is V1 a Cognitive Map? Link to the [2024 GAC website](https://sites.google.com/ccneuro.org/gac2020/gacs-by-year/2024-gacs/2024-1). See the associated [project proposal .pdf](assets/2024_GAC1_Is-V1-a-cognitive-map.pdf).

Data was collected by Dr. Paolo Papale in Dr. Pieter Roelfsema's lab at the Neatherlands Institute for Neuroscience. 

Eye tracking was performed using an OpenIrisDPI digital dual Purkinje image eye tracker, constructed by Ryan Ressmeyer. See [our preprint](https://www.biorxiv.org/content/10.1101/2025.04.18.649589v1) and [our GitHub repository](https://github.com/ryan-ressmeyer/OpenIrisDPI).

Python scripts can be run using the `uv` package manager [link](https://docs.astral.sh/uv/getting-started/installation/). To run the eye movement control script, first set the `datagen_dir` variable then run:

```
1) cp <path to git folder>
2) uv sync
3) uv run scripts/eye_movement_controls.py
```
