# CL2PAKA - Certificateless Two-Party Authenticated Key Agreement

This project is a reference implementation of the **CL2PAKA protocol** in Python.  
It simulates the communication flow between:

- **Trusted Authority (TA)**  
- **Smart Meter (SM)**  
- **Service Provider (SP)**  

Both parties register with the TA and then establish a **shared session key** using the certificateless key agreement protocol.

---

##  Getting Started

- git clone https://github.com/SohamAnwane111/Semester-7.git
- cd NTMC
- python -m venv venv
- venv\Scripts\activate (Windows) | source venv\src\activate (Linux)
- pip install -r requirements.txt
- python .\demo.py