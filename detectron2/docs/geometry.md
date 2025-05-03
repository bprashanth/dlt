# Geometry validation

A valid geometry is one that can be plotted without errors.

## Invalid geometries

1. Self-intersection ("bow-tie") - A geometry that intersects itself.

```
    *----*
     \  /
      \/
      /\
     /  \
    *----*
```

2. Ring self-intersection - when the boundary of a hole in the polygon intersects itself.

```
    *--------*
    |        |
    |  *----*  |
    |  |    |  |
    |  *----*  |
    |        |
    *--------*
```

3. Duplicate vertices - when a polygon has the same vertex more than once.

```
    *----*
    |    |
    |    |
    |    |
    *---**
```

4. Unclosed rings - when the first and last points of a polygon aren't identical
5. Zero area parts - when part of the polygon has no area

## Fixes

Typically these bad geometries are fixed by

1. Adding a small buffer, with the hope that the intersection is resolved.
2. Splitting the polygon into multiple polygons.
3. Removing the offending part (assuming step 2 produces a valid polygon).

So something like this:

```
Before buffer(0):   After buffer(0):
    *----*          *----*
     \  /             \  /
      \/               \/
      /\                /\
     /  \              /  \
    *----*            *----*
```
