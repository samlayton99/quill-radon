### 1-D

Relative $L_2$ on the dense region at $B=128$ (where the meshes differ most):

| task | even | data $p$ | true slope | est. slope | est. curvature | residual | est. frequency |
|---|---|---|---|---|---|---|---|
| 1.1 even grid multiscale bumps | $9\times10^{-14}$ | $2\times10^{-14}$ | $9\times10^{-14}$ | $9\times10^{-14}$ | $8\times10^{-14}$ | $3\times10^{-11}$ | $1\times10^{-13}$ |
| 1.7 uniform data radial runge a12 | $3\times10^{-6}$ | $1\times10^{-7}$ | $2\times10^{-14}$ | $2\times10^{-14}$ | $3\times10^{-14}$ | $3\times10^{-14}$ | $2\times10^{-10}$ |
| 1.8 uniform data sphere jump | $1\times10^{-1}$ | $1\times10^{-1}$ | -- | $1\times10^{-1}$ | $1\times10^{-1}$ | $1\times10^{-1}$ | $2\times10^{-1}$ |
| 1.11 hotspot data radial oscillation freq6 | $3\times10^{-15}$ | $1\times10^{-12}$ | $4\times10^{-14}$ | $4\times10^{-14}$ | $3\times10^{-14}$ | $2\times10^{-13}$ | $1\times10^{-14}$ |
| 1.12 hotspot data packet at hotspot | $8\times10^{-14}$ | $8\times10^{-14}$ | $9\times10^{-14}$ | $9\times10^{-14}$ | $7\times10^{-14}$ | $8\times10^{-14}$ | $2\times10^{-13}$ |
| 1.13 hotspot data packet away from hotspots | $2\times10^{-14}$ | $8\times10^{-11}$ | $6\times10^{-13}$ | $6\times10^{-13}$ | $3\times10^{-14}$ | $2\times10^{-13}$ | $2\times10^{-13}$ |
| 1.14 hotspot data product peak | $1\times10^{-15}$ | $2\times10^{-14}$ | $6\times10^{-15}$ | $5\times10^{-15}$ | $3\times10^{-14}$ | $6\times10^{-14}$ | $1\times10^{-14}$ |
| 1.15 hotspot data jump in sparse region | $2\times10^{-3}$ | $5\times10^{-3}$ | -- | $5\times10^{-3}$ | $2\times10^{-3}$ | $8\times10^{-4}$ | $3\times10^{-3}$ |
| 1.16 hotspot data radial runge a12 at hotspot | $8\times10^{-7}$ | $1\times10^{-9}$ | $2\times10^{-14}$ | $2\times10^{-14}$ | $2\times10^{-14}$ | $1\times10^{-14}$ | $3\times10^{-10}$ |

The same at $B=1024$ (everything resolved; the price of adaptation):

| task | even | data $p$ | true slope | est. slope | est. curvature | residual | est. frequency |
|---|---|---|---|---|---|---|---|
| 1.1 even grid multiscale bumps | $4\times10^{-14}$ | $1\times10^{-13}$ | $1\times10^{-13}$ | $1\times10^{-13}$ | $1\times10^{-13}$ | $2\times10^{-13}$ | $3\times10^{-14}$ |
| 1.7 uniform data radial runge a12 | $7\times10^{-14}$ | $3\times10^{-14}$ | $5\times10^{-13}$ | $5\times10^{-13}$ | $6\times10^{-13}$ | $2\times10^{-13}$ | $4\times10^{-14}$ |
| 1.8 uniform data sphere jump | $5\times10^{-2}$ | $5\times10^{-2}$ | -- | $5\times10^{-1}$ | $2\times10^{-1}$ | $7\times10^{-1}$ | $5\times10^{-2}$ |
| 1.11 hotspot data radial oscillation freq6 | $7\times10^{-14}$ | $5\times10^{-14}$ | $3\times10^{-13}$ | $3\times10^{-13}$ | $2\times10^{-13}$ | $5\times10^{-13}$ | $5\times10^{-14}$ |
| 1.12 hotspot data packet at hotspot | $4\times10^{-14}$ | $3\times10^{-14}$ | $5\times10^{-13}$ | $5\times10^{-13}$ | $8\times10^{-14}$ | $9\times10^{-13}$ | $2\times10^{-14}$ |
| 1.13 hotspot data packet away from hotspots | $5\times10^{-14}$ | $5\times10^{-14}$ | $5\times10^{-14}$ | $5\times10^{-14}$ | $5\times10^{-14}$ | $1\times10^{-13}$ | $1\times10^{-14}$ |
| 1.14 hotspot data product peak | $4\times10^{-14}$ | $3\times10^{-14}$ | $4\times10^{-14}$ | $4\times10^{-14}$ | $5\times10^{-14}$ | $2\times10^{-13}$ | $3\times10^{-14}$ |
| 1.15 hotspot data jump in sparse region | $3\times10^{-5}$ | $2\times10^{-5}$ | -- | $1\times10^{-13}$ | $1\times10^{-12}$ | $7\times10^{-14}$ | $3\times10^{-5}$ |
| 1.16 hotspot data radial runge a12 at hotspot | $6\times10^{-14}$ | $2\times10^{-14}$ | $3\times10^{-13}$ | $3\times10^{-13}$ | $9\times10^{-13}$ | $2\times10^{-13}$ | $3\times10^{-14}$ |

Uniform-cube test at $B=1024$:

| task | even | data $p$ | true slope | est. slope | est. curvature | residual | est. frequency |
|---|---|---|---|---|---|---|---|
| 1.1 even grid multiscale bumps | $2\times10^{-13}$ | $3\times10^{-13}$ | $2\times10^{-13}$ | $2\times10^{-13}$ | $2\times10^{-13}$ | $3\times10^{-11}$ | $2\times10^{-13}$ |
| 1.7 uniform data radial runge a12 | $8\times10^{-14}$ | $3\times10^{-14}$ | $5\times10^{-13}$ | $5\times10^{-13}$ | $6\times10^{-13}$ | $2\times10^{-13}$ | $4\times10^{-14}$ |
| 1.8 uniform data sphere jump | $3\times10^{-2}$ | $3\times10^{-2}$ | -- | $4\times10^{-1}$ | $1\times10^{-1}$ | $4\times10^{-1}$ | $3\times10^{-2}$ |
| 1.11 hotspot data radial oscillation freq6 | $2\times10^{-9}$ | $6\times10^{-12}$ | $2\times10^{-10}$ | $1\times10^{-10}$ | $6\times10^{-10}$ | $5\times10^{-8}$ | $3\times10^{-10}$ |
| 1.12 hotspot data packet at hotspot | $7\times10^{-10}$ | $2\times10^{-12}$ | $2\times10^{-12}$ | $2\times10^{-12}$ | $4\times10^{-12}$ | $8\times10^{-9}$ | $1\times10^{-8}$ |
| 1.13 hotspot data packet away from hotspots | $1\times10^{-9}$ | $7\times10^{-12}$ | $1\times10^{-7}$ | $1\times10^{-7}$ | $7\times10^{-8}$ | $8\times10^{-8}$ | $3\times10^{-7}$ |
| 1.14 hotspot data product peak | $4\times10^{-10}$ | $2\times10^{-12}$ | $3\times10^{-12}$ | $2\times10^{-12}$ | $2\times10^{-10}$ | $2\times10^{-10}$ | $2\times10^{-10}$ |
| 1.15 hotspot data jump in sparse region | $5\times10^{-2}$ | $3\times10^{-2}$ | -- | $2\times10^{-2}$ | $2\times10^{-2}$ | $2\times10^{-2}$ | $4\times10^{0}$ |
| 1.16 hotspot data radial runge a12 at hotspot | $3\times10^{-10}$ | $8\times10^{-13}$ | $5\times10^{-13}$ | $6\times10^{-13}$ | $2\times10^{-12}$ | $1\times10^{-11}$ | $6\times10^{-10}$ |

### 2-D

Dense region at $B=1024$:

| task | even | data $p$ | est. slope | residual | est. frequency | angles (est.) | active (est.) |
|---|---|---|---|---|---|---|---|
| 2.1 even grid multiscale bumps | $2\times10^{-8}$ | $7\times10^{-11}$ | $3\times10^{-9}$ | $3\times10^{-10}$ | $4\times10^{-10}$ | $2\times10^{-8}$ | $3\times10^{-9}$ |
| 2.3 even grid radial oscillation freq6 | $2\times10^{-10}$ | $9\times10^{-10}$ | $3\times10^{-10}$ | $2\times10^{-10}$ | $2\times10^{-10}$ | $2\times10^{-10}$ | $3\times10^{-10}$ |
| 2.7 uniform data radial runge a12 | $5\times10^{-3}$ | $2\times10^{-3}$ | $3\times10^{-3}$ | $3\times10^{-3}$ | $7\times10^{-3}$ | $5\times10^{-3}$ | $3\times10^{-3}$ |
| 2.8 uniform data sphere jump | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ |
| 2.11 hotspot data radial oscillation freq6 | $4\times10^{-11}$ | $2\times10^{-9}$ | $1\times10^{-10}$ | $6\times10^{-11}$ | $4\times10^{-11}$ | $4\times10^{-11}$ | $1\times10^{-10}$ |
| 2.12 hotspot data packet at hotspot | $2\times10^{-4}$ | $2\times10^{-6}$ | $1\times10^{-6}$ | $3\times10^{-5}$ | $1\times10^{-4}$ | $2\times10^{-4}$ | $1\times10^{-6}$ |
| 2.13 hotspot data packet away from hotspots | $2\times10^{-5}$ | $2\times10^{-6}$ | $2\times10^{-6}$ | $3\times10^{-6}$ | $1\times10^{-5}$ | $2\times10^{-5}$ | $2\times10^{-6}$ |
| 2.14 hotspot data product peak | $7\times10^{-5}$ | $7\times10^{-6}$ | $3\times10^{-5}$ | $3\times10^{-5}$ | $6\times10^{-5}$ | $6\times10^{-5}$ | $3\times10^{-5}$ |
| 2.15 curved sheet composition | $7\times10^{-14}$ | $1\times10^{-14}$ | $7\times10^{-14}$ | $8\times10^{-14}$ | $1\times10^{-13}$ | $6\times10^{-14}$ | $7\times10^{-14}$ |
| 2.16 curved sheet noisy composition | $1\times10^{-13}$ | $2\times10^{-13}$ | $2\times10^{-13}$ | $1\times10^{-13}$ | $2\times10^{-13}$ | $1\times10^{-13}$ | $2\times10^{-13}$ |

Dense region at $B=4096$:

| task | even | data $p$ | est. slope | residual | est. frequency | angles (est.) | active (est.) |
|---|---|---|---|---|---|---|---|
| 2.1 even grid multiscale bumps | $2\times10^{-13}$ | $4\times10^{-13}$ | $4\times10^{-13}$ | $2\times10^{-13}$ | $6\times10^{-13}$ | $4\times10^{-13}$ | $4\times10^{-13}$ |
| 2.3 even grid radial oscillation freq6 | $9\times10^{-13}$ | $4\times10^{-11}$ | $2\times10^{-12}$ | $8\times10^{-13}$ | $1\times10^{-12}$ | $9\times10^{-13}$ | $2\times10^{-12}$ |
| 2.7 uniform data radial runge a12 | $7\times10^{-5}$ | $1\times10^{-6}$ | $9\times10^{-7}$ | $9\times10^{-7}$ | $7\times10^{-6}$ | $7\times10^{-5}$ | $9\times10^{-7}$ |
| 2.8 uniform data sphere jump | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ |
| 2.11 hotspot data radial oscillation freq6 | $2\times10^{-13}$ | $3\times10^{-11}$ | $3\times10^{-12}$ | $9\times10^{-13}$ | $3\times10^{-13}$ | $2\times10^{-13}$ | $3\times10^{-12}$ |
| 2.12 hotspot data packet at hotspot | $1\times10^{-11}$ | $2\times10^{-10}$ | $3\times10^{-12}$ | $4\times10^{-13}$ | $1\times10^{-12}$ | $1\times10^{-11}$ | $3\times10^{-12}$ |
| 2.13 hotspot data packet away from hotspots | $6\times10^{-12}$ | $3\times10^{-9}$ | $1\times10^{-12}$ | $5\times10^{-13}$ | $2\times10^{-11}$ | $6\times10^{-12}$ | $1\times10^{-12}$ |
| 2.14 hotspot data product peak | $5\times10^{-8}$ | $1\times10^{-10}$ | $2\times10^{-10}$ | $2\times10^{-10}$ | $5\times10^{-9}$ | $5\times10^{-8}$ | $2\times10^{-10}$ |
| 2.15 curved sheet composition | $3\times10^{-14}$ | $2\times10^{-14}$ | $2\times10^{-14}$ | $2\times10^{-14}$ | $4\times10^{-14}$ | $3\times10^{-14}$ | $2\times10^{-14}$ |
| 2.16 curved sheet noisy composition | $4\times10^{-14}$ | $3\times10^{-14}$ | $4\times10^{-14}$ | $2\times10^{-14}$ | $5\times10^{-14}$ | $3\times10^{-14}$ | $4\times10^{-14}$ |

Uniform-cube test at $B=4096$:

| task | even | data $p$ | est. slope | residual | est. frequency | angles (est.) | active (est.) |
|---|---|---|---|---|---|---|---|
| 2.1 even grid multiscale bumps | $7\times10^{-13}$ | $9\times10^{-13}$ | $9\times10^{-13}$ | $6\times10^{-13}$ | $1\times10^{-12}$ | $9\times10^{-13}$ | $9\times10^{-13}$ |
| 2.3 even grid radial oscillation freq6 | $5\times10^{-12}$ | $1\times10^{-10}$ | $9\times10^{-12}$ | $3\times10^{-12}$ | $3\times10^{-11}$ | $5\times10^{-12}$ | $9\times10^{-12}$ |
| 2.7 uniform data radial runge a12 | $6\times10^{-5}$ | $2\times10^{-6}$ | $2\times10^{-6}$ | $2\times10^{-6}$ | $1\times10^{-5}$ | $6\times10^{-5}$ | $2\times10^{-6}$ |
| 2.8 uniform data sphere jump | $1\times10^{-1}$ | $1\times10^{-1}$ | $1\times10^{-1}$ | $1\times10^{-1}$ | $2\times10^{-1}$ | $1\times10^{-1}$ | $1\times10^{-1}$ |
| 2.11 hotspot data radial oscillation freq6 | $5\times10^{-10}$ | $6\times10^{-9}$ | $8\times10^{-10}$ | $1\times10^{-9}$ | $4\times10^{-10}$ | $5\times10^{-10}$ | $8\times10^{-10}$ |
| 2.12 hotspot data packet at hotspot | $9\times10^{-11}$ | $9\times10^{-8}$ | $9\times10^{-10}$ | $1\times10^{-10}$ | $1\times10^{-9}$ | $9\times10^{-11}$ | $9\times10^{-10}$ |
| 2.13 hotspot data packet away from hotspots | $4\times10^{-10}$ | $2\times10^{-6}$ | $1\times10^{-9}$ | $3\times10^{-9}$ | $6\times10^{-8}$ | $3\times10^{-10}$ | $1\times10^{-9}$ |
| 2.14 hotspot data product peak | $2\times10^{-6}$ | $8\times10^{-9}$ | $4\times10^{-8}$ | $2\times10^{-8}$ | $2\times10^{-7}$ | $2\times10^{-6}$ | $4\times10^{-8}$ |
| 2.15 curved sheet composition | $5\times10^{-1}$ | $6\times10^{-1}$ | $6\times10^{-1}$ | $6\times10^{-1}$ | $4\times10^{-1}$ | $5\times10^{-1}$ | $6\times10^{-1}$ |
| 2.16 curved sheet noisy composition | $2\times10^{-1}$ | $4\times10^{-1}$ | $3\times10^{-1}$ | $3\times10^{-1}$ | $1\times10^{-1}$ | $2\times10^{-1}$ | $3\times10^{-1}$ |

### $d=3$ and $d=5$ at $B=4096$

Dense region:

| task | even | data $p$ | est. slope | est. frequency | active (true) | active (est.) | active, iterated |
|---|---|---|---|---|---|---|---|
| 3.5 uniform data composition | $4\times10^{-7}$ | $2\times10^{-7}$ | $3\times10^{-7}$ | -- | -- | $2\times10^{-11}$ | $2\times10^{-11}$ |
| 3.7 uniform data radial runge a12 | $2\times10^{-2}$ | $2\times10^{-2}$ | $2\times10^{-2}$ | -- | -- | $2\times10^{-2}$ | $2\times10^{-2}$ |
| 3.11 hotspot data radial oscillation freq6 | $2\times10^{-8}$ | $6\times10^{-8}$ | $3\times10^{-8}$ | -- | -- | $3\times10^{-8}$ | $3\times10^{-8}$ |
| 3.12 hotspot data packet at hotspot | $4\times10^{-2}$ | $3\times10^{-2}$ | $4\times10^{-2}$ | -- | -- | $4\times10^{-2}$ | $4\times10^{-2}$ |
| 3.13 hotspot data packet away from hotspots | $7\times10^{-3}$ | $5\times10^{-3}$ | $6\times10^{-3}$ | -- | -- | $6\times10^{-3}$ | $6\times10^{-3}$ |
| 3.16 curved sheet noisy composition | $8\times10^{-9}$ | $1\times10^{-9}$ | $3\times10^{-9}$ | -- | -- | $2\times10^{-13}$ | $2\times10^{-13}$ |
| 5.5 uniform data composition | $8\times10^{-4}$ | $8\times10^{-4}$ | $8\times10^{-4}$ | -- | -- | $5\times10^{-6}$ | $5\times10^{-6}$ |
| 5.16 curved sheet noisy composition | $1\times10^{-6}$ | $1\times10^{-6}$ | $1\times10^{-6}$ | -- | -- | $5\times10^{-11}$ | $5\times10^{-11}$ |

Uniform-cube test:

| task | even | data $p$ | est. slope | est. frequency | active (true) | active (est.) | active, iterated |
|---|---|---|---|---|---|---|---|
| 3.5 uniform data composition | $9\times10^{-7}$ | $4\times10^{-7}$ | $7\times10^{-7}$ | -- | -- | $8\times10^{-10}$ | $8\times10^{-10}$ |
| 3.7 uniform data radial runge a12 | $3\times10^{-2}$ | $2\times10^{-2}$ | $3\times10^{-2}$ | -- | -- | $3\times10^{-2}$ | $3\times10^{-2}$ |
| 3.11 hotspot data radial oscillation freq6 | $3\times10^{-5}$ | $5\times10^{-5}$ | $3\times10^{-5}$ | -- | -- | $3\times10^{-5}$ | $3\times10^{-5}$ |
| 3.12 hotspot data packet at hotspot | $4\times10^{0}$ | $3\times10^{0}$ | $3\times10^{0}$ | -- | -- | $3\times10^{0}$ | $3\times10^{0}$ |
| 3.13 hotspot data packet away from hotspots | $8\times10^{-1}$ | $8\times10^{-1}$ | $5\times10^{-1}$ | -- | -- | $5\times10^{-1}$ | $4\times10^{-1}$ |
| 3.16 curved sheet noisy composition | $4\times10^{-1}$ | $1\times10^{-1}$ | $2\times10^{-1}$ | -- | -- | $3\times10^{-2}$ | $3\times10^{-2}$ |
| 5.5 uniform data composition | $2\times10^{-3}$ | $2\times10^{-3}$ | $2\times10^{-3}$ | -- | -- | $3\times10^{-2}$ | $3\times10^{-2}$ |
| 5.16 curved sheet noisy composition | $1\times10^{0}$ | $2\times10^{0}$ | $2\times10^{0}$ | -- | -- | $9\times10^{-2}$ | $9\times10^{-2}$ |

### $d=3$ split sweep (even mesh, $B=4096$), dense region

| task | 8 per dir | 12 per dir | 16 per dir | 24 per dir | 32 per dir | 48 per dir | 64 per dir |
|---|---|---|---|---|---|---|---|
| 3.3 even grid radial oscillation freq6 | $9\times10^{-3}$ | $4\times10^{-5}$ | $2\times10^{-7}$ | $1\times10^{-7}$ | $9\times10^{-6}$ | $7\times10^{-4}$ | $6\times10^{-3}$ |
| 3.7 uniform data radial runge a12 | $8\times10^{-2}$ | $4\times10^{-2}$ | $2\times10^{-2}$ | $1\times10^{-2}$ | $2\times10^{-2}$ | $4\times10^{-2}$ | $7\times10^{-2}$ |
| 3.11 hotspot data radial oscillation freq6 | $5\times10^{-4}$ | $2\times10^{-6}$ | $2\times10^{-8}$ | $8\times10^{-9}$ | $6\times10^{-7}$ | $5\times10^{-5}$ | $4\times10^{-4}$ |
| 3.12 hotspot data packet at hotspot | $5\times10^{-1}$ | $2\times10^{-1}$ | $4\times10^{-2}$ | $8\times10^{-3}$ | $2\times10^{-2}$ | $6\times10^{-2}$ | $8\times10^{-2}$ |
| 3.13 hotspot data packet away from hotspots | $6\times10^{-2}$ | $2\times10^{-2}$ | $7\times10^{-3}$ | $5\times10^{-3}$ | $1\times10^{-2}$ | $1\times10^{-2}$ | $2\times10^{-2}$ |
| 3.16 curved sheet noisy composition | $2\times10^{-6}$ | $1\times10^{-7}$ | $8\times10^{-9}$ | $3\times10^{-11}$ | $2\times10^{-11}$ | $3\times10^{-10}$ | $9\times10^{-9}$ |

### Known-answer ridge in 2-D (uniform test)

| $B$ | even angles | angles from $A(\theta)^{1/3}$ | active (true) | active (est.) | active, iterated |
|---|---|---|---|---|---|
| 256 | $2\times10^{-2}$ | $1\times10^{-2}$ | $2\times10^{1}$ | $2\times10^{1}$ | $7\times10^{4}$ |
| 512 | $6\times10^{-5}$ | $6\times10^{-5}$ | $7\times10^{0}$ | $7\times10^{0}$ | $5\times10^{5}$ |
| 1024 | $7\times10^{-6}$ | $8\times10^{-6}$ | $4\times10^{-2}$ | $4\times10^{-2}$ | $1\times10^{4}$ |
| 2048 | $1\times10^{-7}$ | $1\times10^{-7}$ | $2\times10^{-2}$ | $2\times10^{-2}$ | $3\times10^{2}$ |
| 4096 | $3\times10^{-10}$ | $4\times10^{-10}$ | $3\times10^{-3}$ | $3\times10^{-3}$ | $5\times10^{1}$ |