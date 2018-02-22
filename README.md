# pyVideoPartialCopyDetector
Proof-of-concept implementation of a Partial-Video-Copy-Detector implemented in Python (and some C++)

This is a python-based prototype to compare two videos for partial-overlapping in a robust way.

### Algorithmics
- Let *ffmpeg* extract the *most representative frame* in each time-interval (e.g. 1s)
    - This idea somewhat is based on file-deduplication approaches like a *rolling-hash* for *fingerprinting*
    - Example-usage would be:
        - ```ffmpeg -i "clipA.mkv" -vf thumbnail=25 -vf fps=1 DATA/A/%04d.jpg```
        - ```ffmpeg -i "clipB.mkv" -vf thumbnail=25 -vf fps=1 DATA/B/%04d.jpg```

- *Perceptual-hashing of images*: two algorithms used together
    - Statistics-based:
        - ```"From Image Hashing to Video Hashing" / Li Weng and Bart Preneel```
    - Block-Histogram-based
        - Various papers
- Min-cost-flow-based Temporal-alignment
    - ```"Scalable detection of partial near-duplicate videos by visual-temporal consistency / Tan, Hung-Khoon, et al.```

### Implementation
- The statistics-based hasher is implemented using *cython* to improve performance.
- The min-cost-flow-based temporal-alignment is prepared with networkx, but solved through CoinOR's Lemon (C++)
    - This dependency ```pyLemonFlow``` is based on *pybind11* and available in another repository

### Demo
When comparing two different movie-trailers, which have different intros and outros, also
slightly different visuals due to *english vs. german*, the code can visualize the following:

- the first row are the frames of clip 1
- the third row are the frames of clip 3
- the *middle-row* are the selected frames of clip2, matched to 1

![Demo](https://i.imgur.com/i2Dzqsa.jpg)

### Remarks
The code is just a prototype with hidden hyper-parameters. It's incomplete for real-world usage
and performance is still an issue (for larger videos).
