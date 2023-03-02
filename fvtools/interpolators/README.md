# Base-classes for interpolators

# Nearest4
`nearest4` computes bi-linear horizontal interpolation

For a structured grid arranged like so:
```
    |       |
    |       |
--- n ----- n ----
    |       |
    |       |
    |   m   |
    | f     |
    |       |
--- n ----- n ----
    |       |
    |       |
```
it finds the 4 `n` points that form a box around each `m` point. We thereafter find the closest `m` point to each FVCOM point `f`, and find the bi-linear interpolation coefficients for data from `n` points to `f` points.
