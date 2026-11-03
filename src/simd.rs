/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! CPU capabilities discovered once while constructing an encoder.
//!
//! Private flags can only come from detection. SIMD selectors reuse this
//! snapshot, and encoding uses the resulting function pointers throughout.

#[derive(Clone, Copy)]
pub(crate) struct CpuFeatures {
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    avx2: bool,
    #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
    sse2: bool,
    #[cfg(all(
        feature = "polish",
        feature = "sse",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    sse41: bool,
}

impl CpuFeatures {
    pub(crate) fn detect() -> Self {
        // NEON is baseline on AArch64 and needs no runtime probe.
        Self {
            #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
            avx2: std::is_x86_feature_detected!("avx2"),
            #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
            sse2: std::is_x86_feature_detected!("sse2"),
            #[cfg(all(
                feature = "polish",
                feature = "sse",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            sse41: std::is_x86_feature_detected!("sse4.1"),
        }
    }

    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    pub(crate) fn avx2(self) -> bool {
        self.avx2
    }

    #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
    pub(crate) fn sse2(self) -> bool {
        self.sse2
    }

    #[cfg(all(
        feature = "polish",
        feature = "sse",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    pub(crate) fn sse41(self) -> bool {
        self.sse41
    }
}
