/*!
Frontend for [WGSL][wgsl] (WebGPU Shading Language).

[wgsl]: https://gpuweb.github.io/gpuweb/wgsl.html
*/

mod error;
mod index;
mod lower;
mod parse;
#[cfg(test)]
mod tests;

pub use crate::front::wgsl::error::ParseError;
pub use crate::front::wgsl::parse::Options;
pub use crate::front::wgsl::parse::directive::language_extension::{
    ImplementedLanguageExtension, LanguageExtension, UnimplementedLanguageExtension,
};

use alloc::boxed::Box;
use alloc::vec::Vec;
use thiserror::Error;

use crate::Scalar;
use crate::front::wgsl::error::Error;
use crate::front::wgsl::lower::Lowerer;
use crate::front::wgsl::parse::Parser;

#[cfg(test)]
use std::println;

pub(crate) type Result<'a, T> = core::result::Result<T, Vec<Box<Error<'a>>>>;

pub struct Frontend {
    parser: Parser,
    options: Options,
}

impl Frontend {
    pub const fn new() -> Self {
        Self {
            parser: Parser::new(),
            options: Options::new(),
        }
    }
    pub const fn new_with_options(options: Options) -> Self {
        Self {
            parser: Parser::new(),
            options,
        }
    }

    pub fn parse<'a>(
        &mut self,
        source: &'a str,
    ) -> core::result::Result<crate::Module, ParseError<'a>> {
        self.inner(source).map_err(|errors| {
            // Convert Vec<Box<Error>> to a single ParseError containing all errors
            ParseError::Multiple(errors)
        })
    }

    fn inner<'a>(&mut self, source: &'a str) -> Result<'a, crate::Module> {
        let tu = match self.parser.parse(source, &self.options) {
            Ok(tu) => tu,
            Err(errors) => return Err(errors),
        };
        let index = index::Index::generate(&tu)?;
        let module = Lowerer::new(&index).lower(tu)?;

        Ok(module)
    }
}

/// <div class="warning">
// NOTE: Keep this in sync with `wgpu::Device::create_shader_module`!
// NOTE: Keep this in sync with `wgpu_core::Global::device_create_shader_module`!
///
/// This function may consume a lot of stack space. Compiler-enforced limits for parsing recursion
/// exist; if shader compilation runs into them, it will return an error gracefully. However, on
/// some build profiles and platforms, the default stack size for a thread may be exceeded before
/// this limit is reached during parsing. Callers should ensure that there is enough stack space
/// for this, particularly if calls to this method are exposed to user input.
///
/// </div>
pub fn parse_str(source: &str) -> core::result::Result<crate::Module, ParseError<'_>> {
    let mut frontend = Frontend::new();
    frontend.parse(source)
}

#[cfg(test)]
#[track_caller]
pub fn assert_parse_err(input: &str, snapshot: &str) {
    let output = parse_str(input)
        .expect_err("expected parser error")
        .emit_to_string(input);
    if output != snapshot {
        for diff in diff::lines(snapshot, &output) {
            match diff {
                diff::Result::Left(l) => println!("-{l}"),
                diff::Result::Both(l, _) => println!(" {l}"),
                diff::Result::Right(r) => println!("+{r}"),
            }
        }
        panic!("Error snapshot failed");
    }
}
