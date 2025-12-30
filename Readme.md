# Interior Wall Segmentation

A simple Tkinter-based desktop application that uses Meta’s Segment Anything Model 3 (SAM 3) to automatically detect and segment interior painted walls in room photographs.

Built by [@Guhanbala](https://github.com/Guhanbala)

---

## Features

- Shadow-aware preprocessing for better wall detection in real-world interior images  
- Text-prompted segmentation using SAM 3 with the prompt: **"interior painted wall"**  
- Automatic cleaning of wall masks  
- Rectification of wall boundaries by snapping edges to horizontal and vertical directions  
- Interactive selection: click near a wall to highlight it  
- Semi-transparent colored overlays with numbered wall regions  

---

## Demo Screenshot

![App Screenshot](screenshot.png)

---

## Requirements

- Python 3.8 or higher  
- NVIDIA GPU with CUDA recommended (CPU execution supported but slower)  
- SAM 3 model checkpoint (`sam3.pt`) — gated access required  

---

## Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/Guhanbala/interior-wall-segmentation.git
cd interior-wall-segmentation
````

### Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

* Linux / macOS

  ```bash
  source venv/bin/activate
  ```
* Windows

  ```bash
  venv\Scripts\activate
  ```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download SAM 3 Checkpoint

1. Visit: [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. Request access (gated)
3. Download `sam3.pt` after approval
4. Place `sam3.pt` in the project root directory

Optional (after approval):

```bash
huggingface-cli download facebook/sam3 sam3.pt --local-dir .
```

---

## Run the Application

```bash
python wall_segmentation_app.py
```

---

## Project Structure

```text
interior-wall-segmentation/
├── wall_segmentation_app.py
├── sam3.pt
├── requirements.txt
├── screenshot.png
└── README.md
```

---

## Notes

* Uses text-prompt-based segmentation (no manual annotation)
* SAM 3 is used without fine-tuning
* Designed specifically for interior painted walls
