## Steps to Get the script running


1. Open Command Promnt : 
    `nvidia-smi` , note the GPU Name and CUDA Version and Driver Version
2. Goto https://www.nvidia.com/en-us/drivers/ and input the details given above. Install the latest driver.
3. Restart the system
4. run `nvcc --version`, if showing something then ok if not :<br>
    a. goto https://developer.nvidia.com/cuda-12-1-0-download-archive
    b. Fill out the details 
    c. Select .exe(Local)
    d. Install it, and run `nvcc --version` again.
5. Install Python if not already installed
6. Open terminal and type :
    `python -m venv venv`<br>
    `.\venv\Scripts\activate`
7. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
8. `pip install numpy opencv-python-headless scikit-learn ultralytics`
9. C:\doc_processor\
│
├── MOCK_DATA\
│   └── documents\
│       ├── your_image_1.png
│       └── your_image_2.jpg
│
├── source\
│   └── docLayout.pt
│
├── run.py              <-- (Your main script, MODIFIED for "cuda")
├── deskew.py           <-- (Your deskew helper script)
└── doclayout_yolo.py   <-- [!] This file must exist here
Make sure the layout is like this

10. python run.py
