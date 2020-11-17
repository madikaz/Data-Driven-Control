# How to

## How to use pv_pannel

### Initialization:
```python
from pv_solver import pv_panel
panel = pv_panel(Np = 10, Ns = 10)
```
Np denotes the number of cells connected in parallel and Ns denotes the number connected in series. Defaults are set to be 10

### Functions:
1. giving the voltage, solve for current
    ```
    I,d = panel.solve_i(G, T, I, V)
    ```
    Inputs:
    - G: irradiance
    - T: temperature
    - I: initial guess of the current
    - V: given voltage

    Outputs:
    - I: current giving the voltage
    - d: dI/dV at the given point

2. giving the current, solve for voltage
    ```
    V,d = panel.solve_v(G, T, I, V)
    ```
    Inputs:
    - G: irradiance
    - T: temperature
    - I: given current
    - V: initial guess of the voltage

    Outputs:
    - V: voltage giving the current
    - d: dI/dV at the given point

3. find the maximum power
    ```
    P_mpp, V_mpp = panel.find_opt(G,T)
    ```
    Inputs:
    - G: irradiance
    - T: Temperature

    Outputs:
    - P_mpp: maximum power
    - V_mpp: operating voltage at P_MPP

4. find the reference power point
    ```
    P_ref, V_ref = panel.find_vref(G,T, p_ref)
    ```
    Inputs:
    - G: irradiance
    - T: Temperature
    - p_ref: reference power

    Outputs:
    - P_ref: the actual reference power, min{P_MPP, p_ref}
    - V_ref: operating voltage at P_ref

## How to use sos_tracker

### Initialization:
```python
from pv_solver import sos_tracker

tracker = sos_tracker(buffer_size=24, tolerance=12, eps=1e-9, deg=6, norm_factor=50, freq=10)
```

### Major functions:

1. update the curve giving the measured data I,V

    ```
    tracker.update_curve(I, V)
    ```
    Inputs:
    - I: a list of recently measured current
    - V: a list of recently measured voltage one to one matches with the input current

    No output.
    
    This function should be called before used for other estimation

2. find the maximum power
    ```
    P_mpp, V_mpp = tracker.find_opt(v_est)
    ```
    Inputs:
    - v_est: initial guess of the maximum power point

    Outputs:
    - P_mpp: maximum power
    - V_mpp: operating voltage at P_MPP

3. find the reference power point
    ```
    P_ref, V_ref = tracker.find_vref(p_ref, v_ref, clipping = True)
    ```
    Inputs:
    - p_ref: reference power
    - v_ref: operating voltage at last time 
    - clipping: whether to clip the operating voltage based on the range of the history values for stability

    Outputs:
    - P_ref: the actual reference power, min{P_MPP, p_ref}
    - V_ref: operating voltage at P_ref

### How to use irradiance:

### Initialization:
```python
from utils import irradiance

ird = irradiance(path, length, offset=0, row = 1, interpolate = 'cubic')
```

arguments:
- path: csv file path
- length: the data length (second) we want to use 
- offset: the start time (in second)
- row: can be 1 to 5, corresponding to 5 groups of irradiance data
- interpolate: what kind of interpolate method we want to use for sub-second time period (can be 'cubic', 'linear', 'nearist', 'previous')

### query data

There are 3 equivilant ways to query the irradiance data:
```
G = ird[t]
G = ird(t)
G = ird.get_ird(t)
```

Input:
- t: float, the time of the iradiance data, need to be less than the length 

Output:
- G: irradiance at query time t

You can use the following ways to
 ```
 len(ird)
 ird.get_length()
 ird.length
 ```
to check the length of the irradiance data