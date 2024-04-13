---
title: Background
---
In this section, we first go over some key terms and definitions that are helpful to know when using CADDEE. Next, we briefly discuss CADDEE's design philosophy and introduce the key (native) classes that a user will interacts with. This includes a briew description of the main classes as well as a UML class diagram that shows the relationship between classes.

## Terms and definitions
- System: *A machine that manages power to perform tasks that involve motion and forces* (Ex: aircraft, car, wind turbine, robot)
- Component: *A subset of the system without which it cannot perform its intended task* (Ex: wing, engine, blade, actuator)
- Geometry: *A spatial representation of a component, typically in the form of a mathematical function*
- Free-form deformation (FFD): *A parameterization method for manipulating the geometry of one or multiple components that involves parametrically defining a complex geometry as a function of a simple geometry such as a block.*
- Configuration: *A variation of the system with a set of discrete or continuous changes to one or more of its components compared to the base configuration* (Ex: aircraft with deflected control surface, retracted landing gear, removed/added winglets or even entire wing, different number of engines or batteries, etc.)
- (Design) condition: *A description of a mode of operation for the system, usually defined by rigid-body states.* (Ex: cruise condition)
    - On-design condition: *A description of an intended  mode of operation of the system* (Ex: cruise condition)
    - Off-design condition: *A description of an un-intended mode of operation of the system* (Ex: one-engine inoperative (OEI), structural sizing conditions like +3g pull-up maneuver)
- Mesh
- Solver: *A computer program that implements a computational model* (Ex: VAST, ADFlow, FlowUnsteady, lsdo_rotor, CCBlade)
    - Computational model: *A set of explicit computations resulting from applying numerical solvers, like linear/non-linear solvers, to numerical models*
        - Numerical model: *A set of algebraic equations resulting from the discretization of differential or integral operators found in a mthematical model*
            - Mathematical model: *A set of equations (that may include differential or integral operators) whose solution describes the predicted behavior or performance of a system* (Ex: steady RANS equations)


## CADDEE design philosphy and classes

### Design philosophy

CADDEE is implemented in Python and subscribes to the **object-oriented programming (OOP)** paradigm, which allows a user to operate at a high level of abstraction. This is particularly convenient when constructing complex computational models where a user is only interested in high-level outputs and inputs and how they are used to connect solvers. Solvers that are integrated through CADDEE are written in the **Computational System Design Language (CSDL)**, a python-based language for multidisciplinary design optimization (MDO) whose critical feature is **automated sensitivity analysis** for gradient-based optimization. Unlike CADDEE, CSDL subsribes to the functional programming (FP) paradigm, which allows the developers of solvers to precisely and unambiguously define the mathemtical operations that make up the computational model that the solver implements. For assembling large computatational models, a user will treat solvers like a black box that performs analyses and outputs certain quantities of interest that CADDEE will organize and proecess. 

The design philosophy of CADDEE is centered around *simplicity*, *flexibility*, and *extensibility*.
- Simplicity: CADDEE is a light-weight package that aims to enable the construction of complex computational models through few, well-designed classes
- Flexibility: CADDEE is amenable to various combinations of design conditions, configurations and components. As such, changes to the system can be defined through different configurations and analyzed across different conditions. For example, variations of certain types of air taxi concepts can be easily constructed by adding or removing rotors, batteries or other components and analyzing these configurations in different design conditions, like a fast or slow cruise. 
- Extensibility: While conceived with aircraft design in mind, CADDEE is designed extend to other disciplines like wind-turbines or robotics.

The following desscription of CADDEE's classes aims to illustrate some of the aspects of its design philosophy.

### General description of classes

There are four main classes native to the CADDEE framework: 
1. `CADDEE`
2. `Condition`
3. `Configuration`
4. `Component`. 

At the top level, the `CADDEE` class acts as container of two objects, the base configuration (instance of `Configuration`) and  a conditions dictionary. The conditions dictionary stores instances of the second major class, `Condition`. Conditions serve as a way to compartmentalize the analysis (done through discipline solvers) of the engineering system. Subclasses of `Condition`, like `AircraftCondition` narrow the scope of the analysis to a specific application domain (i.e., aircraft design) and have sepcific condition parameters, like mach number, altitude, or flight path angle. The base `Condition` class has two attributes: *quantities*, which is a dictionary used to store certain (user-specified) quantities of interest and *configuration*, which is an instance of the `Configuration` class. Typically, a condition contains axactly one configuration, which is a different instance than the base configuration of the `CADDEE` class.

The `Configuration` class is the third major class in CADDEE and serves several key functionalities. 

### UML class diagram

The unified modeling language is the standard visual modeling language for system and software engineering, intended for constructing, visualizing, and documenting the design of (software) systems. It provids a standard set of diagrams and symbols for communicating the design and architecutre of a software framework. The following fiugre provides a brief overiew of some of the basic UML symbols that we use in the class diagram below. Inheritance relationship are shown primarily as examples of domain-specific sub-classes (aircraft-design in this cases) and are not exhausitve.

```{figure} /src/images/uml_notation.png
:figwidth: 100 %
:align: center
:alt: caddee_classes

```
These UML symbols have the following meaning
- Association: A relationship between two or more classes. It indicates that objects of one class are connected or interact with objects of another class. Associations can be binary (between two classes), or they can involve more than two classes.
- Composition: A strong form of association that represents a whole-part relationship, indicating that the parts (contained objects) cannot exist independently of the whole (container object).
- Aggregation: A weaker form of association where the contained objects can exist independently of the container object. It represents a "has-a" relationship, indicating that the contained objects can belong to one or multiple container objects or exist on their own.
- Dependency: A directed relationship where a UML element depends on or requires another element to be specified or implemented. 
- Inheritance: A parent-child relationship where the child inherits the attributes and mathods from the parent.

In addition, the symbols `+` and `-` are used to denote public and private class attributes and methods, respectively. The CADDEE class diagram is shown below, where the navy-blue color indicates classes that are natively in CADDEE while the gold color indicates classes that imported into CADDEE.

```{figure} /src/images/CADDEE_classes.png
:figwidth: 90 %
:align: center
:alt: caddee_classes
:target: ../_images/CADDEE_classes.png

```

