# jhm17

A JPEG encoder written in Rust, forked from
[vstroebel/jpeg-encoder](https://github.com/vstroebel/jpeg-encoder)

## Example
```rust
use jhm17::{Encoder, ColorType};

// An array with 4 pixels in RGB format.
let data = [
    255, 0, 0,
    0, 255, 0,
    0, 0, 255,
    255, 255, 255,
];

// Create new encoder that writes to a file with maximum quality (100)
let mut encoder = Encoder::new_file("some.jpeg", 100)?;

// Encode the data with dimension 2x2
encoder.encode(&data, 2, 2, ColorType::Rgb)?;
```

## License

Code written for jhm17 is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE))

at your option.

This software is based in part on the work of the Independent JPEG Group;
see [LICENSE-IJG](LICENSE-IJG) for the derived portions, including the arithmetic
coder adapted from IJG/libjpeg-turbo.
