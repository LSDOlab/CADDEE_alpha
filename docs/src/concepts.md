---
title: Theory and Concepts
---

In this section,  we cover some key concepts, including the reference frames and rigid-body states that CADDEE uses for analyses. This is important informations for users as well as developers that wish to interface their solvers with CADDEE. 


## Reference frames and rigid-body states
There are currently two main reference frames used in CADDEE, both of which follow the right-hand rule convention. 

1. Inertial reference frame (denoted with subscipt "I"): fixed to the ground with z-axis pointing "up"
2. Body-fixed reference frame (denoted with subscript "B"): fixed to the system, which is allowed to translate and rotate relative to the inertial reference frame. 

In the figures below, we show an isometric and several 2-D views of the body-fixed (and inertial) reference frame for the case of aircrfat flight-dynamics. By convention, the body-fixed reference frame for aircraft is defined as
- X: in the direction of the nose of the aircraft
- Y: in the direction of the right wing 
- Z: downward 

Related to the concept of the inertial and body-fixed reference frame are the rigid-body states (also referred to as "aircraft states" in the aircraft design community) that CADDEE uses, which are given by the following 12-dimensional state vector $\mathbf{s}$

<center>

$\mathbf{s} = [u, v, w, p, q, r, \phi, \theta, \psi, x, y, z]$,

</center>

where $\mathbf{u} = [u, v, w]$ are the linear velocities, $\boldsymbol{\omega} = [p, q, r]$ are rotation rates (i.e., angular velocities), and $[\phi, \theta, \psi]$ are the Euler angles corresponding to roll, pitch, and yaw, respectively. These quantities are all taken with respect to the body-fixed reference frame. Lastly, $\mathbf{x} = [x, y, z]$ are the coordinates of the system (i.e., the origin of the body-fixed frame) with respect to the inertial reference frame. 

### Isometric view

```{figure} /src/images/isometric_view.jpg
:figwidth: 100 %
:align: center
:alt: caddee_classes
:target: ../_images/isometric_view.jpg

```


### 2-D views

<div style="width:400px">Top view</div>    |  <div style="width:290px">Front view</div>        |  <div style="width:290px">Side view</div> |
| :---:     |    :---:         |   :---:    |
![](/src/images/top_view.jpg)  |  ![](/src/images/front_view.jpg)   |  ![](/src/images/side_view.jpg)|

