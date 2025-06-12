# Thesis Project: Reconstructing legal
3D apartment models from
2D division drawings

## Overview
This research focused on converting 2D legal division drawings of apartment buildings into 3D digital models. These 3D models represent the legal ownership structure of buildings, which parts are private apartments and which are shared spaces.

In the Netherlands, apartment ownership is recorded through deeds and 2D drawings. However, not all buildings have detailed 3D BIM (Building Information Model) data available. This project provides a way to automatically generate 3D BIM Legal models from existing 2D division drawings, even when no original BIM exists.

What the project does:
- Uses pre-vectorized 2D division drawings as input
- Aligns the drawings to real-world coordinates (from the BGT)
- Estimates storey heights and aligns floors vertically
- Builds simple 3D representations of the building’s legal layout
- Outputs the result in CityJSON format, following the BIM Legal LoD1+ standard

These models help visualize how ownership is divided in a building in 3D, which can be useful for legal registration, urban planning, and future automation of property documentation.
![image2](https://github.com/user-attachments/assets/41822074-d712-483a-815a-2847740e5037)

## Project Structure

```
thesis/
├── data/
│   ├── raw/           # Original input data (e.g., shapefiles)
│   └── json_input/    # Vectorized JSONs of division drawings
│
├── output/            # Results (.city.json files)
├── main.py            # Main script to run
└── README.md
```

The code starts at line 1016. Nothing has to be adjusted for it to run.
You can choose to turn off "automatic capping", this will allow you to manually set the height of each storey that is beyond reasonable limits (2.1-3.3 m).
You can also turn on "manual", which will allow you to manually adjust georeferencing parameters.
Initialization techniques can also be changed, but they have been set to the best performing methods.

This project is part of my MSc thesis at Geomatics TU Delft.
Supervised by: Jantien Stoter, Amir Hakim, Ken Arroyo Ohori and Bhavya Kausika
Data provided by: Kadaster
