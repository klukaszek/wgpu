//! Formatting WGSL front end error messages.

use crate::common::wgsl::TryToWgsl;
use crate::diagnostic_filter::ConflictingDiagnosticRuleError;
use crate::proc::{Alignment, ConstantEvaluatorError, ResolveError};
use crate::{Scalar, SourceLocation, Span};

use super::parse::directive::enable_extension::{EnableExtension, UnimplementedEnableExtension};
use super::parse::directive::language_extension::{
    LanguageExtension, UnimplementedLanguageExtension,
};
use super::parse::lexer::Token;

use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term;
use thiserror::Error;

use alloc::{
    borrow::Cow,
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::ops::Range;

#[derive(Clone, Debug)]
pub enum ParseError<'a> {
    Single {
        message: String,
        // The first span should be the primary span, and the other ones should be complementary.
        labels: Vec<(Span, Cow<'static, str>)>,
        notes: Vec<String>,
    },
    Multiple(Vec<Box<Error<'a>>>),
}

impl<'a> ParseError<'a> {
    pub fn labels(&self) -> Box<dyn Iterator<Item = (Span, &str)> + '_> {
        match self {
            ParseError::Single { labels, .. } => {
                Box::new(labels.iter().map(|&(span, ref msg)| (span, msg.as_ref())))
            }
            ParseError::Multiple(_) => Box::new(std::iter::empty()),
        }
    }

    pub fn message(&self) -> &str {
        match self {
            ParseError::Single { message, .. } => message,
            ParseError::Multiple(_) => "Multiple parse errors",
        }
    }

    fn diagnostic(&self) -> Diagnostic<()> {
        match self {
            ParseError::Single {
                message,
                labels,
                notes,
            } => Diagnostic::error()
                .with_message(message.to_string())
                .with_labels(
                    labels
                        .iter()
                        .filter_map(|label| label.0.to_range().map(|range| (label, range)))
                        .map(|(label, range)| {
                            Label::primary((), range).with_message(label.1.to_string())
                        })
                        .collect(),
                )
                .with_notes(notes.iter().map(|note| format!("note: {note}")).collect()),
            ParseError::Multiple(_) => Diagnostic::error().with_message("Multiple parse errors"),
        }
    }

    /// Emits a summary of the error to standard error stream.
    #[cfg(feature = "stderr")]
    pub fn emit_to_stderr(&self, source: &str) {
        match self {
            ParseError::Single { .. } => self.emit_to_stderr_with_path(source, "wgsl"),
            ParseError::Multiple(errors) => {
                for error in errors {
                    error.emit_to_stderr(source);
                }
            }
        }
    }

    /// Emits a summary of the error to standard error stream.
    #[cfg(feature = "stderr")]
    pub fn emit_to_stderr_with_path<P>(&self, source: &str, path: P)
    where
        P: AsRef<std::path::Path>,
    {
        match self {
            ParseError::Single { .. } => {
                let path = path.as_ref().display().to_string();
                let files = SimpleFile::new(path, source);
                let config = term::Config::default();

                cfg_if::cfg_if! {
                    if #[cfg(feature = "termcolor")] {
                        let writer = term::termcolor::StandardStream::stderr(term::termcolor::ColorChoice::Auto);
                    } else {
                        let writer = std::io::stderr();
                    }
                }

                term::emit(&mut writer.lock(), &config, &files, &self.diagnostic())
                    .expect("cannot write error");
            }
            ParseError::Multiple(errors) => {
                for error in errors {
                    error.emit_to_stderr_with_path(source, &path);
                }
            }
        }
    }

    /// Emits a summary of the error to a string.
    pub fn emit_to_string(&self, source: &str) -> String {
        match self {
            ParseError::Single { .. } => self.emit_to_string_with_path(source, "wgsl"),
            ParseError::Multiple(errors) => {
                let mut output = String::new();
                for error in errors {
                    output.push_str(&error.emit_to_string(source));
                    output.push('\n');
                }
                output
            }
        }
    }

    /// Emits a summary of the error to a string.
    pub fn emit_to_string_with_path<P>(&self, source: &str, path: P) -> String
    where
        P: AsRef<std::path::Path>,
    {
        match self {
            ParseError::Single { .. } => {
                let path = path.as_ref().display().to_string();
                let files = SimpleFile::new(path, source);
                let config = term::Config::default();

                let mut writer = crate::error::DiagnosticBuffer::new();
                term::emit(writer.inner_mut(), &config, &files, &self.diagnostic())
                    .expect("cannot write error");
                writer.into_string()
            }
            ParseError::Multiple(errors) => {
                let mut output = String::new();
                for error in errors {
                    output.push_str(&error.emit_to_string_with_path(source, &path));
                    output.push('\n');
                }
                output
            }
        }
    }

    /// Returns a [`SourceLocation`] for the first label in the error message.
    pub fn location(&self, source: &str) -> Option<SourceLocation> {
        match self {
            ParseError::Single { labels, .. } => {
                labels.first().map(|label| label.0.location(source))
            }
            ParseError::Multiple(errors) => errors.first().and_then(|e| e.location(source)),
        }
    }
}

impl<'a> From<Vec<Box<Error<'a>>>> for ParseError<'a> {
    fn from(errors: Vec<Box<Error<'a>>>) -> Self {
        ParseError::Multiple(errors)
    }
}

impl<'a> core::fmt::Display for ParseError<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ParseError::Single { message, .. } => write!(f, "{}", message),
            ParseError::Multiple(errors) => {
                for error in errors {
                    writeln!(f, "{}", error)?;
                }
                Ok(())
            }
        }
    }
}

impl<'a> core::error::Error for ParseError<'a> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        None
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ExpectedToken<'a> {
    Token(Token<'a>),
    Identifier,
    AfterIdentListComma,
    AfterIdentListArg,
    /// Expected: constant, parenthesized expression, identifier
    PrimaryExpression,
    /// Expected: assignment, increment/decrement expression
    Assignment,
    /// Expected: 'case', 'default', '}'
    SwitchItem,
    /// Expected: ',', ')'
    WorkgroupSizeSeparator,
    /// Expected: 'struct', 'let', 'var', 'type', ';', 'fn', eof
    GlobalItem,
    /// Expected a type.
    Type,
    /// Access of `var`, `let`, `const`.
    Variable,
    /// Access of a function
    Function,
    /// The `diagnostic` identifier of the `@diagnostic(…)` attribute.
    DiagnosticAttribute,
}

#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum NumberError {
    #[error("invalid numeric literal format")]
    Invalid,
    #[error("numeric literal not representable by target type")]
    NotRepresentable,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum InvalidAssignmentType {
    Other,
    Swizzle,
    ImmutableBinding(Span),
}

impl<'a> Error<'a> {
    pub fn emit_to_string(&self, source: &str) -> String {
        use codespan_reporting::diagnostic::Diagnostic;
        use codespan_reporting::files::SimpleFile;
        use codespan_reporting::term;
        let diagnostic = Diagnostic::error().with_message(format!("{}", self));
        let files = SimpleFile::new("wgsl", source);
        let config = term::Config::default();
        let mut writer = crate::error::DiagnosticBuffer::new();
        term::emit(writer.inner_mut(), &config, &files, &diagnostic).expect("cannot write error");
        writer.into_string()
    }

    /// Returns a reference to the span if present, otherwise None.
    pub fn span(&self) -> Option<&Span> {
        match self {
            Error::Unexpected(span, _)
            | Error::UnexpectedComponents(span)
            | Error::UnexpectedOperationInConstContext(span)
            | Error::BadNumber(span, _)
            | Error::BadMatrixScalarKind(span, _)
            | Error::BadAccessor(span)
            | Error::BadTexture(span)
            | Error::NotStorageTexture(span)
            | Error::BadIncrDecrReferenceType(span)
            | Error::InvalidForInitializer(span)
            | Error::InvalidBreakIf(span)
            | Error::InvalidGatherComponent(span)
            | Error::InvalidConstructorComponentType(span, _)
            | Error::InvalidIdentifierUnderscore(span)
            | Error::ReservedIdentifierPrefix(span)
            | Error::UnknownAddressSpace(span)
            | Error::RepeatedAttribute(span)
            | Error::UnknownAttribute(span)
            | Error::UnknownBuiltin(span)
            | Error::UnknownAccess(span)
            | Error::UnknownIdent(span, _)
            | Error::UnknownScalarType(span)
            | Error::UnknownType(span)
            | Error::UnknownStorageFormat(span)
            | Error::UnknownConservativeDepth(span)
            | Error::UnknownEnableExtension(span, _)
            | Error::UnknownLanguageExtension(span, _)
            | Error::UnknownDiagnosticRuleName(span)
            | Error::SizeAttributeTooLow(span, _)
            | Error::AlignAttributeTooLow(span, _)
            | Error::NonPowerOfTwoAlignAttribute(span)
            | Error::InconsistentBinding(span)
            | Error::TypeNotConstructible(span)
            | Error::TypeNotInferable(span)
            | Error::DeclMissingTypeAndInit(span)
            | Error::MissingAttribute(_, span)
            | Error::InvalidAddrOfOperand(span)
            | Error::InvalidAtomicPointer(span)
            | Error::InvalidAtomicOperandType(span)
            | Error::InvalidRayQueryPointer(span)
            | Error::NotPointer(span)
            | Error::NotReference(_, span)
            | Error::InvalidAssignment { span, .. }
            | Error::ReservedKeyword(span)
            | Error::Redefinition { previous: span, .. }
            | Error::RecursiveDeclaration { ident: span, .. }
            | Error::CyclicDeclaration { ident: span, .. }
            | Error::InvalidSwitchSelector { span }
            | Error::InvalidSwitchCase { span }
            | Error::SwitchCaseTypeMismatch { span }
            | Error::CalledEntryPoint(span)
            | Error::WrongArgumentCount { span, .. }
            | Error::TooManyArguments {
                call_span: span, ..
            }
            | Error::WrongArgumentType {
                call_span: span, ..
            }
            | Error::FunctionReturnsVoid(span)
            | Error::ExpectedConstExprConcreteIntegerScalar(span)
            | Error::ExpectedNonNegative(span)
            | Error::ExpectedPositiveArrayLength(span)
            | Error::MissingWorkgroupSize(span)
            | Error::ExceededLimitForNestedBraces { span, .. }
            | Error::PipelineConstantIDValue(span)
            | Error::EnableExtensionNotYetImplemented { span, .. }
            | Error::EnableExtensionNotEnabled { span, .. }
            | Error::LanguageExtensionNotYetImplemented { span, .. }
            | Error::DiagnosticInvalidSeverity {
                severity_control_name_span: span,
            }
            | Error::SelectUnexpectedArgumentType { arg_span: span, .. }
            | Error::SelectRejectAndAcceptHaveNoCommonType {
                reject_span: span, ..
            } => Some(span),
            _ => None,
        }
    }

    pub fn emit_to_string_with_path<P>(&self, source: &str, path: P) -> String
    where
        P: AsRef<std::path::Path>,
    {
        use codespan_reporting::diagnostic::Diagnostic;
        use codespan_reporting::files::SimpleFile;
        use codespan_reporting::term;
        let diagnostic = Diagnostic::error().with_message(format!("{}", self));
        let path = path.as_ref().display().to_string();
        let files = SimpleFile::new(path, source);
        let config = term::Config::default();
        let mut writer = crate::error::DiagnosticBuffer::new();
        term::emit(writer.inner_mut(), &config, &files, &diagnostic).expect("cannot write error");
        writer.into_string()
    }

    pub fn location(&self, source: &str) -> Option<SourceLocation> {
        // For now, just return None (could be improved to extract from error variants)
        None
    }
}

impl<'a> core::fmt::Display for Error<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use Error::*;
        match self {
            Unexpected(span, expected) => {
                write!(f, "Unexpected token at {:?}, expected {:?}", span, expected)
            }
            UnexpectedComponents(span) => write!(f, "Unexpected components at {:?}", span),
            UnexpectedOperationInConstContext(span) => {
                write!(f, "Unexpected operation in const context at {:?}", span)
            }
            BadNumber(span, err) => write!(f, "Bad number at {:?}: {:?}", span, err),
            BadMatrixScalarKind(span, scalar) => {
                write!(f, "Bad matrix scalar kind at {:?}: {:?}", span, scalar)
            }
            BadAccessor(span) => write!(f, "Bad accessor at {:?}", span),
            BadTexture(span) => write!(f, "Bad texture at {:?}", span),
            BadTypeCast {
                span,
                from_type,
                to_type,
            } => write!(
                f,
                "Bad type cast at {:?}: {} to {}",
                span, from_type, to_type
            ),
            NotStorageTexture(span) => write!(f, "Not a storage texture at {:?}", span),
            BadTextureSampleType { span, scalar } => {
                write!(f, "Bad texture sample type at {:?}: {:?}", span, scalar)
            }
            BadIncrDecrReferenceType(span) => {
                write!(f, "Bad increment/decrement reference type at {:?}", span)
            }
            InvalidResolve(err) => write!(f, "Invalid resolve: {:?}", err),
            InvalidForInitializer(span) => write!(f, "Invalid for initializer at {:?}", span),
            InvalidBreakIf(span) => write!(f, "Invalid break if at {:?}", span),
            InvalidGatherComponent(span) => write!(f, "Invalid gather component at {:?}", span),
            InvalidConstructorComponentType(span, idx) => write!(
                f,
                "Invalid constructor component type at {:?}, index {}",
                span, idx
            ),
            InvalidIdentifierUnderscore(span) => {
                write!(f, "Invalid identifier underscore at {:?}", span)
            }
            ReservedIdentifierPrefix(span) => write!(f, "Reserved identifier prefix at {:?}", span),
            UnknownAddressSpace(span) => write!(f, "Unknown address space at {:?}", span),
            RepeatedAttribute(span) => write!(f, "Repeated attribute at {:?}", span),
            UnknownAttribute(span) => write!(f, "Unknown attribute at {:?}", span),
            UnknownBuiltin(span) => write!(f, "Unknown builtin at {:?}", span),
            UnknownAccess(span) => write!(f, "Unknown access at {:?}", span),
            UnknownIdent(span, ident) => write!(f, "Unknown identifier '{}' at {:?}", ident, span),
            UnknownScalarType(span) => write!(f, "Unknown scalar type at {:?}", span),
            UnknownType(span) => write!(f, "Unknown type at {:?}", span),
            UnknownStorageFormat(span) => write!(f, "Unknown storage format at {:?}", span),
            UnknownConservativeDepth(span) => write!(f, "Unknown conservative depth at {:?}", span),
            UnknownEnableExtension(span, ext) => {
                write!(f, "Unknown enable extension '{}' at {:?}", ext, span)
            }
            UnknownLanguageExtension(span, ext) => {
                write!(f, "Unknown language extension '{}' at {:?}", ext, span)
            }
            UnknownDiagnosticRuleName(span) => {
                write!(f, "Unknown diagnostic rule name at {:?}", span)
            }
            SizeAttributeTooLow(span, size) => {
                write!(f, "Size attribute too low at {:?}: {}", span, size)
            }
            AlignAttributeTooLow(span, align) => {
                write!(f, "Align attribute too low at {:?}: {:?}", span, align)
            }
            NonPowerOfTwoAlignAttribute(span) => {
                write!(f, "Non-power-of-two align attribute at {:?}", span)
            }
            InconsistentBinding(span) => write!(f, "Inconsistent binding at {:?}", span),
            TypeNotConstructible(span) => write!(f, "Type not constructible at {:?}", span),
            TypeNotInferable(span) => write!(f, "Type not inferable at {:?}", span),
            InitializationTypeMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "Initialization type mismatch for '{}' at {:?}: expected {}, got {}",
                expected, name, expected, got
            ),
            DeclMissingTypeAndInit(span) => {
                write!(f, "Declaration missing type and initializer at {:?}", span)
            }
            MissingAttribute(attr, span) => write!(f, "Missing attribute '{}' at {:?}", attr, span),
            InvalidAddrOfOperand(span) => write!(f, "Invalid address-of operand at {:?}", span),
            InvalidAtomicPointer(span) => write!(f, "Invalid atomic pointer at {:?}", span),
            InvalidAtomicOperandType(span) => {
                write!(f, "Invalid atomic operand type at {:?}", span)
            }
            InvalidRayQueryPointer(span) => write!(f, "Invalid ray query pointer at {:?}", span),
            NotPointer(span) => write!(f, "Not a pointer at {:?}", span),
            NotReference(what, span) => write!(f, "Not a reference to {} at {:?}", what, span),
            InvalidAssignment { span, ty } => {
                write!(f, "Invalid assignment at {:?}: {:?}", span, ty)
            }
            ReservedKeyword(span) => write!(f, "Reserved keyword at {:?}", span),
            _ => write!(f, "Error: {:?}", self),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Error<'a> {
    Unexpected(Span, ExpectedToken<'a>),
    UnexpectedComponents(Span),
    UnexpectedOperationInConstContext(Span),
    BadNumber(Span, NumberError),
    BadMatrixScalarKind(Span, Scalar),
    BadAccessor(Span),
    BadTexture(Span),
    BadTypeCast {
        span: Span,
        from_type: String,
        to_type: String,
    },
    NotStorageTexture(Span),
    BadTextureSampleType {
        span: Span,
        scalar: Scalar,
    },
    BadIncrDecrReferenceType(Span),
    InvalidResolve(ResolveError),
    InvalidForInitializer(Span),
    /// A break if appeared outside of a continuing block
    InvalidBreakIf(Span),
    InvalidGatherComponent(Span),
    InvalidConstructorComponentType(Span, i32),
    InvalidIdentifierUnderscore(Span),
    ReservedIdentifierPrefix(Span),
    UnknownAddressSpace(Span),
    RepeatedAttribute(Span),
    UnknownAttribute(Span),
    UnknownBuiltin(Span),
    UnknownAccess(Span),
    UnknownIdent(Span, &'a str),
    UnknownScalarType(Span),
    UnknownType(Span),
    UnknownStorageFormat(Span),
    UnknownConservativeDepth(Span),
    UnknownEnableExtension(Span, &'a str),
    UnknownLanguageExtension(Span, &'a str),
    UnknownDiagnosticRuleName(Span),
    SizeAttributeTooLow(Span, u32),
    AlignAttributeTooLow(Span, Alignment),
    NonPowerOfTwoAlignAttribute(Span),
    InconsistentBinding(Span),
    TypeNotConstructible(Span),
    TypeNotInferable(Span),
    InitializationTypeMismatch {
        name: Span,
        expected: String,
        got: String,
    },
    DeclMissingTypeAndInit(Span),
    MissingAttribute(&'static str, Span),
    InvalidAddrOfOperand(Span),
    InvalidAtomicPointer(Span),
    InvalidAtomicOperandType(Span),
    InvalidRayQueryPointer(Span),
    NotPointer(Span),
    NotReference(&'static str, Span),
    InvalidAssignment {
        span: Span,
        ty: InvalidAssignmentType,
    },
    ReservedKeyword(Span),
    /// Redefinition of an identifier (used for both module-scope and local redefinitions).
    Redefinition {
        /// Span of the identifier in the previous definition.
        previous: Span,

        /// Span of the identifier in the new definition.
        current: Span,
    },
    /// A declaration refers to itself directly.
    RecursiveDeclaration {
        /// The location of the name of the declaration.
        ident: Span,

        /// The point at which it is used.
        usage: Span,
    },
    /// A declaration refers to itself indirectly, through one or more other
    /// definitions.
    CyclicDeclaration {
        /// The location of the name of some declaration in the cycle.
        ident: Span,

        /// The edges of the cycle of references.
        ///
        /// Each `(decl, reference)` pair indicates that the declaration whose
        /// name is `decl` has an identifier at `reference` whose definition is
        /// the next declaration in the cycle. The last pair's `reference` is
        /// the same identifier as `ident`, above.
        path: Box<[(Span, Span)]>,
    },
    InvalidSwitchSelector {
        span: Span,
    },
    InvalidSwitchCase {
        span: Span,
    },
    SwitchCaseTypeMismatch {
        span: Span,
    },
    CalledEntryPoint(Span),
    WrongArgumentCount {
        span: Span,
        expected: Range<u32>,
        found: u32,
    },
    /// No overload of this function accepts this many arguments.
    TooManyArguments {
        /// The name of the function being called.
        function: String,

        /// The function name in the call expression.
        call_span: Span,

        /// The first argument that is unacceptable.
        arg_span: Span,

        /// Maximum number of arguments accepted by any overload of
        /// this function.
        max_arguments: u32,
    },
    /// A value passed to a builtin function has a type that is not
    /// accepted by any overload of the function.
    WrongArgumentType {
        /// The name of the function being called.
        function: String,

        /// The function name in the call expression.
        call_span: Span,

        /// The first argument whose type is unacceptable.
        arg_span: Span,

        /// The index of the first argument whose type is unacceptable.
        arg_index: u32,

        /// That argument's actual type.
        arg_ty: String,

        /// The set of argument types that would have been accepted for
        /// this argument, given the prior arguments.
        allowed: Vec<String>,
    },
    /// A value passed to a builtin function has a type that is not
    /// accepted, given the earlier arguments' types.
    InconsistentArgumentType {
        /// The name of the function being called.
        function: String,

        /// The function name in the call expression.
        call_span: Span,

        /// The first unacceptable argument.
        arg_span: Span,

        /// The index of the first unacceptable argument.
        arg_index: u32,

        /// The actual type of the first unacceptable argument.
        arg_ty: String,

        /// The prior argument whose type made the `arg_span` argument
        /// unacceptable.
        inconsistent_span: Span,

        /// The index of the `inconsistent_span` argument.
        inconsistent_index: u32,

        /// The type of the `inconsistent_span` argument.
        inconsistent_ty: String,

        /// The types that would have been accepted instead of the
        /// first unacceptable argument.
        allowed: Vec<String>,
    },
    FunctionReturnsVoid(Span),
    FunctionMustUseUnused(Span),
    FunctionMustUseReturnsVoid(Span, Span),
    InvalidWorkGroupUniformLoad(Span),
    Internal(&'static str),
    ExpectedConstExprConcreteIntegerScalar(Span),
    ExpectedNonNegative(Span),
    ExpectedPositiveArrayLength(Span),
    MissingWorkgroupSize(Span),
    ConstantEvaluatorError(Box<ConstantEvaluatorError>, Span),
    AutoConversion(Box<AutoConversionError>),
    AutoConversionLeafScalar(Box<AutoConversionLeafScalarError>),
    ConcretizationFailed(Box<ConcretizationFailedError>),
    ExceededLimitForNestedBraces {
        span: Span,
        limit: u8,
    },
    PipelineConstantIDValue(Span),
    NotBool(Span),
    ConstAssertFailed(Span),
    DirectiveAfterFirstGlobalDecl {
        directive_span: Span,
    },
    EnableExtensionNotYetImplemented {
        kind: UnimplementedEnableExtension,
        span: Span,
    },
    EnableExtensionNotEnabled {
        kind: EnableExtension,
        span: Span,
    },
    LanguageExtensionNotYetImplemented {
        kind: UnimplementedLanguageExtension,
        span: Span,
    },
    DiagnosticInvalidSeverity {
        severity_control_name_span: Span,
    },
    DiagnosticDuplicateTriggeringRule(ConflictingDiagnosticRuleError),
    DiagnosticAttributeNotYetImplementedAtParseSite {
        site_name_plural: &'static str,
        spans: Vec<Span>,
    },
    DiagnosticAttributeNotSupported {
        on_what: DiagnosticAttributeNotSupportedPosition,
        spans: Vec<Span>,
    },
    SelectUnexpectedArgumentType {
        arg_span: Span,
        arg_type: String,
    },
    SelectRejectAndAcceptHaveNoCommonType {
        reject_span: Span,
        reject_type: String,
        accept_span: Span,
        accept_type: String,
    },
}

impl From<ConflictingDiagnosticRuleError> for Error<'_> {
    fn from(value: ConflictingDiagnosticRuleError) -> Self {
        Self::DiagnosticDuplicateTriggeringRule(value)
    }
}

/// Used for diagnostic refinement in [`Error::DiagnosticAttributeNotSupported`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum DiagnosticAttributeNotSupportedPosition {
    SemicolonInModulePosition,
    Other { display_plural: &'static str },
}

impl From<&'static str> for DiagnosticAttributeNotSupportedPosition {
    fn from(display_plural: &'static str) -> Self {
        Self::Other { display_plural }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AutoConversionError {
    pub dest_span: Span,
    pub dest_type: String,
    pub source_span: Span,
    pub source_type: String,
}

#[derive(Clone, Debug)]
pub(crate) struct AutoConversionLeafScalarError {
    pub dest_span: Span,
    pub dest_scalar: String,
    pub source_span: Span,
    pub source_type: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ConcretizationFailedError {
    pub expr_span: Span,
    pub expr_type: String,
    pub scalar: String,
    pub inner: ConstantEvaluatorError,
}

impl<'a> Error<'a> {
    #[cold]
    #[inline(never)]
    pub(crate) fn as_parse_error(&self, source: &'a str) -> ParseError {
        match self {
            &Error::Unexpected(unexpected_span, expected) => {
                let expected_str = match expected {
                    ExpectedToken::Token(token) => match token {
                        Token::Separator(c) => format!("`{c}`"),
                        Token::Paren(c) => format!("`{c}`"),
                        Token::Attribute => "@".to_string(),
                        Token::Number(_) => "number".to_string(),
                        Token::Word(s) => s.to_string(),
                        Token::Operation(c) => format!("operation (`{c}`)"),
                        Token::LogicalOperation(c) => format!("logical operation (`{c}`)"),
                        Token::ShiftOperation(c) => format!("bitshift (`{c}{c}`)"),
                        Token::AssignmentOperation(c) if c == '<' || c == '>' => {
                            format!("bitshift (`{c}{c}=`)")
                        }
                        Token::AssignmentOperation(c) => format!("operation (`{c}=`)"),
                        Token::IncrementOperation => "increment operation".to_string(),
                        Token::DecrementOperation => "decrement operation".to_string(),
                        Token::Arrow => "->".to_string(),
                        Token::Unknown(c) => format!("unknown (`{c}`)"),
                        Token::Trivia => "trivia".to_string(),
                        Token::DocComment(s) => format!("doc comment ('{s}')"),
                        Token::ModuleDocComment(s) => format!("module doc comment ('{s}')"),
                        Token::End => "end".to_string(),
                    },
                    ExpectedToken::Identifier => "identifier".to_string(),
                    ExpectedToken::PrimaryExpression => "expression".to_string(),
                    ExpectedToken::Assignment => "assignment or increment/decrement".to_string(),
                    ExpectedToken::SwitchItem => concat!(
                        "switch item (`case` or `default`) or a closing curly bracket ",
                        "to signify the end of the switch statement (`}`)"
                    )
                    .to_string(),
                    ExpectedToken::WorkgroupSizeSeparator => {
                        "workgroup size separator (`,`) or a closing parenthesis".to_string()
                    }
                    ExpectedToken::GlobalItem => concat!(
                        "global item (`struct`, `const`, `var`, `alias`, ",
                        "`fn`, `diagnostic`, `enable`, `requires`, `;`) ",
                        "or the end of the file"
                    )
                    .to_string(),
                    ExpectedToken::Type => "type".to_string(),
                    ExpectedToken::Variable => "variable access".to_string(),
                    ExpectedToken::Function => "function name".to_string(),
                    ExpectedToken::AfterIdentListArg => {
                        "next argument, trailing comma, or end of list (`,` or `;`)".to_string()
                    }
                    ExpectedToken::AfterIdentListComma => {
                        "next argument or end of list (`;`)".to_string()
                    }
                    ExpectedToken::DiagnosticAttribute => {
                        "the `diagnostic` attribute identifier".to_string()
                    }
                };
                ParseError::Single {
                    message: format!(
                        "expected {}, found {:?}",
                        expected_str, &source[unexpected_span],
                    ),
                    labels: vec![(unexpected_span, format!("expected {expected_str}").into())],
                    notes: vec![],
                }
            }
            &Error::UnexpectedComponents(bad_span) => ParseError::Single {
                message: "unexpected components".to_string(),
                labels: vec![(bad_span, "unexpected components".into())],
                notes: vec![],
            },
            &Error::UnexpectedOperationInConstContext(span) => ParseError::Single {
                message: "this operation is not supported in a const context".to_string(),
                labels: vec![(span, "operation not supported here".into())],
                notes: vec![],
            },
            &Error::BadNumber(bad_span, ref err) => ParseError::Single {
                message: format!("{}: `{}`", err, &source[bad_span],),
                labels: vec![(bad_span, err.to_string().into())],
                notes: vec![],
            },
            &Error::BadMatrixScalarKind(span, scalar) => ParseError::Single {
                message: format!(
                    "matrix scalar type must be floating-point, but found `{}`",
                    scalar.to_wgsl_for_diagnostics()
                ),
                labels: vec![(span, "must be floating-point (e.g. `f32`)".into())],
                notes: vec![],
            },
            &Error::BadAccessor(bad_span) => ParseError::Single {
                message: format!("invalid field accessor `{}`", &source[bad_span],),
                labels: vec![(bad_span, "invalid accessor".into())],
                notes: vec![],
            },
            &Error::UnknownIdent(ident_span, ident) => ParseError::Single {
                message: format!("no definition in scope for identifier: `{ident}`"),
                labels: vec![(ident_span, "unknown identifier".into())],
                notes: vec![],
            },
            &Error::UnknownScalarType(bad_span) => ParseError::Single {
                message: format!("unknown scalar type: `{}`", &source[bad_span]),
                labels: vec![(bad_span, "unknown scalar type".into())],
                notes: vec!["Valid scalar types are f32, f64, i32, u32, bool".into()],
            },
            &Error::NotStorageTexture(bad_span) => ParseError::Single {
                message: "textureStore can only be applied to storage textures".to_string(),
                labels: vec![(bad_span, "not a storage texture".into())],
                notes: vec![],
            },
            &Error::BadTextureSampleType { span, scalar } => ParseError::Single {
                message: format!(
                    "texture sample type must be one of f32, i32 or u32, but found {}",
                    scalar.to_wgsl_for_diagnostics()
                ),
                labels: vec![(span, "must be one of f32, i32 or u32".into())],
                notes: vec![],
            },
            &Error::BadIncrDecrReferenceType(span) => ParseError::Single {
                message: concat!(
                    "increment/decrement operation requires ",
                    "a reference type of i32 or u32"
                )
                .to_string(),
                labels: vec![(span, "must be a reference type of i32 or u32".into())],
                notes: vec![],
            },
            &Error::BadTexture(bad_span) => ParseError::Single {
                message: format!(
                    "expected an image, but found `{}` which is not an image",
                    &source[bad_span]
                ),
                labels: vec![(bad_span, "not an image".into())],
                notes: vec![],
            },
            &Error::BadTypeCast {
                span,
                ref from_type,
                ref to_type,
            } => {
                let msg = format!("cannot cast a {from_type} to a {to_type}");
                ParseError::Single {
                    message: msg.clone(),
                    labels: vec![(span, msg.into())],
                    notes: vec![],
                }
            }
            &Error::InvalidResolve(ref resolve_error) => ParseError::Single {
                message: resolve_error.to_string(),
                labels: vec![],
                notes: vec![],
            },
            &Error::InvalidForInitializer(bad_span) => ParseError::Single {
                message: format!(
                    "for(;;) initializer is not an assignment or a function call: `{}`",
                    &source[bad_span]
                ),
                labels: vec![(bad_span, "not an assignment or function call".into())],
                notes: vec![],
            },
            &Error::InvalidBreakIf(bad_span) => ParseError::Single {
                message: "break-if is only allowed in continuing block".to_string(),
                labels: vec![(bad_span, "not in a continuing block".into())],
                notes: vec![],
            },
            &Error::InvalidGatherComponent(bad_span) => ParseError::Single {
                message: format!(
                    "textureGather component `{}` doesn't exist, must be 0, 1, 2, or 3",
                    &source[bad_span]
                ),
                labels: vec![(bad_span, "invalid component".into())],
                notes: vec![],
            },
            &Error::InvalidConstructorComponentType(bad_span, component) => ParseError::Single {
                message: format!("invalid type for constructor component at index [{component}]"),
                labels: vec![(bad_span, "invalid component type".into())],
                notes: vec![],
            },
            &Error::InvalidIdentifierUnderscore(bad_span) => ParseError::Single {
                message: "Identifier can't be `_`".to_string(),
                labels: vec![(bad_span, "invalid identifier".into())],
                notes: vec![
                    "Use phony assignment instead (`_ =` notice the absence of `let` or `var`)"
                        .to_string(),
                ],
            },
            &Error::ReservedIdentifierPrefix(bad_span) => ParseError::Single {
                message: format!(
                    "Identifier starts with a reserved prefix: `{}`",
                    &source[bad_span]
                ),
                labels: vec![(bad_span, "invalid identifier".into())],
                notes: vec![],
            },
            &Error::UnknownAddressSpace(span) => ParseError::Single {
                message: "unknown address space".to_string(),
                labels: vec![(span, "unknown address space".into())],
                notes: vec![],
            },
            &Error::RepeatedAttribute(span) => ParseError::Single {
                message: "repeated attribute".to_string(),
                labels: vec![(span, "repeated attribute".into())],
                notes: vec![],
            },
            &Error::UnknownAttribute(span) => ParseError::Single {
                message: "unknown attribute".to_string(),
                labels: vec![(span, "unknown attribute".into())],
                notes: vec![],
            },
            &Error::UnknownBuiltin(span) => ParseError::Single {
                message: "unknown builtin".to_string(),
                labels: vec![(span, "unknown builtin".into())],
                notes: vec![],
            },
            &Error::UnknownAccess(span) => ParseError::Single {
                message: "unknown access".to_string(),
                labels: vec![(span, "unknown access".into())],
                notes: vec![],
            },
            &Error::UnknownType(span) => ParseError::Single {
                message: "unknown type".to_string(),
                labels: vec![(span, "unknown type".into())],
                notes: vec![],
            },
            &Error::UnknownStorageFormat(span) => ParseError::Single {
                message: "unknown storage format".to_string(),
                labels: vec![(span, "unknown storage format".into())],
                notes: vec![],
            },
            &Error::UnknownConservativeDepth(span) => ParseError::Single {
                message: "unknown conservative depth".to_string(),
                labels: vec![(span, "unknown conservative depth".into())],
                notes: vec![],
            },
            &Error::UnknownEnableExtension(span, word) => ParseError::Single {
                message: format!("unknown enable-extension `{}`", word),
                labels: vec![(span, "".into())],
                notes: vec![
                    "See available extensions at <https://www.w3.org/TR/WGSL/#enable-extension>."
                        .into(),
                ],
            },
            &Error::Redefinition { previous, current } => ParseError::Single {
                message: format!("redefinition of identifier"),
                labels: vec![
                    (previous, "previous definition".into()),
                    (current, "redefinition".into()),
                ],
                notes: vec![],
            },
            &Error::RecursiveDeclaration { ident, usage } => ParseError::Single {
                message: "recursive declaration".to_string(),
                labels: vec![
                    (ident, "declaration".into()),
                    (usage, "uses itself here".into()),
                ],
                notes: vec![],
            },
            &Error::CyclicDeclaration { ident, ref path } => ParseError::Single {
                message: "cyclic declaration".to_string(),
                labels: {
                    let mut labels = vec![(ident, "cycle starts here".into())];
                    labels.extend(path.iter().map(|(decl, _reference)| (*decl, "cycle".into())));
                    labels
                },
                notes: vec![],
            },
            &Error::InvalidSwitchSelector { span } => ParseError::Single {
                message: "invalid switch selector".to_string(),
                labels: vec![(span, "invalid selector".into())],
                notes: vec![],
            },
            &Error::InvalidSwitchCase { span } => ParseError::Single {
                message: "invalid switch case".to_string(),
                labels: vec![(span, "invalid case".into())],
                notes: vec![],
            },
            &Error::SwitchCaseTypeMismatch { span } => ParseError::Single {
                message: "switch case type mismatch".to_string(),
                labels: vec![(span, "type mismatch".into())],
                notes: vec![],
            },
            &Error::UnknownLanguageExtension(span, name) => ParseError::Single {
                message: format!("unknown language extension `{name}`"),
                labels: vec![(span, "".into())],
                notes: vec![concat!(
                    "See available extensions at ",
                    "<https://www.w3.org/TR/WGSL/#language-extensions-sec>."
                )
                .into()],
            },
            &Error::UnknownDiagnosticRuleName(span) => ParseError::Single {
                message: format!("unknown `diagnostic(…)` rule name `{}`", &source[span]),
                labels: vec![(span, "not a valid diagnostic rule name".into())],
                notes: vec![concat!(
                    "See available trigger rules at ",
                    "<https://www.w3.org/TR/WGSL/#filterable-triggering-rules>."
                )
                .into()],
            },
            &Error::SizeAttributeTooLow(bad_span, min_size) => ParseError::Single {
                message: format!("struct member size must be at least {min_size}"),
                labels: vec![(bad_span, format!("must be at least {min_size}").into())],
                notes: vec![],
            },
            &Error::AlignAttributeTooLow(bad_span, min_align) => ParseError::Single {
                message: format!("struct member alignment must be at least {min_align}"),
                labels: vec![(bad_span, format!("must be at least {min_align}").into())],
                notes: vec![],
            },
            &Error::NonPowerOfTwoAlignAttribute(bad_span) => ParseError::Single {
                message: "struct member alignment must be a power of 2".to_string(),
                labels: vec![(bad_span, "must be a power of 2".into())],
                notes: vec![],
            },
            &Error::InconsistentBinding(span) => ParseError::Single {
                message: "input/output binding is not consistent".to_string(),
                labels: vec![(span, "input/output binding is not consistent".into())],
                notes: vec![],
            },
            &Error::CalledEntryPoint(span) => ParseError::Single {
                message: "entry point cannot be called".to_string(),
                labels: vec![(span, "entry point cannot be called".into())],
                notes: vec![],
            },
            &Error::WrongArgumentCount { span, ref expected, found } => ParseError::Single {
                message: format!("wrong number of arguments: expected {:?}, found {}", expected, found),
                labels: vec![(span, "wrong number of arguments".into())],
                notes: vec![],
            },
            &Error::TooManyArguments { ref function, call_span, arg_span, max_arguments } => ParseError::Single {
                message: format!("too many arguments passed to `{}`", function),
                labels: vec![
                    (call_span, "".into()),
                    (arg_span, format!("unexpected argument #{}", max_arguments + 1).into())
                ],
                notes: vec![
                    format!("The `{}` function accepts at most {} argument(s)", function, max_arguments)
                ],
            },
            &Error::WrongArgumentType { ref function, call_span, arg_span, arg_index, ref arg_ty, ref allowed } => {
                let message = format!(
                    "wrong type passed as argument #{} to `{}`",
                    arg_index + 1,
                    function
                );
                let labels = vec![
                    (call_span, "".into()),
                    (arg_span, format!("argument #{} has type `{}`", arg_index + 1, arg_ty).into())
                ];
                let mut notes = vec![];
                notes.push(format!("`{}` accepts the following types for argument #{}:", function, arg_index + 1));
                notes.extend(allowed.iter().map(|ty| format!("allowed type: {}", ty)));
                ParseError::Single { message, labels, notes }
            },
            &Error::TypeNotConstructible(span) => ParseError::Single {
                message: "type is not constructible".to_string(),
                labels: vec![(span, "type is not constructible".into())],
                notes: vec![],
            },
            &Error::TypeNotInferable(span) => ParseError::Single {
                message: "type not inferable".to_string(),
                labels: vec![(span, "type not inferable".into())],
                notes: vec![],
            },
            &Error::InitializationTypeMismatch {
                name,
                ref expected,
                ref got,
            } => ParseError::Single {
                message: format!(
                    "the type of `{}` is expected to be `{}`, but got `{}`",
                    &source[name], expected, got,
                ),
                labels: vec![(name, format!("definition of `{}`", &source[name]).into())],
                notes: vec![],
            },
            &Error::DeclMissingTypeAndInit(name_span) => ParseError::Single {
                message: format!(
                    "declaration of `{}` needs a type specifier or initializer",
                    &source[name_span]
                ),
                labels: vec![(name_span, "needs a type specifier or initializer".into())],
                notes: vec![],
            },
            &Error::MissingAttribute(name, name_span) => ParseError::Single {
                message: format!(
                    "variable `{}` needs a '{}' attribute",
                    &source[name_span], name
                ),
                labels: vec![(
                    name_span,
                    format!("definition of `{}`", &source[name_span]).into(),
                )],
                notes: vec![],
            },
            &Error::InvalidAddrOfOperand(span) => ParseError::Single {
                message: "invalid operand for address-of operator".to_string(),
                labels: vec![(span, "invalid operand for address-of".into())],
                notes: vec![],
            },
            &Error::InvalidAtomicPointer(span) => ParseError::Single {
                message: "atomic operation is done on a pointer to a non-atomic".to_string(),
                labels: vec![(span, "atomic pointer is invalid".into())],
                notes: vec![],
            },
            &Error::InvalidAtomicOperandType(span) => ParseError::Single {
                message: "atomic operand type is inconsistent with the operation".to_string(),
                labels: vec![(span, "atomic operand type is invalid".into())],
                notes: vec![],
            },
            &Error::InvalidRayQueryPointer(span) => ParseError::Single {
                message: "ray query operation is done on a pointer to a non-ray-query".to_string(),
                labels: vec![(span, "ray query pointer is invalid".into())],
                notes: vec![],
            },
            &Error::NotPointer(span) => ParseError::Single {
                message: "the operand of the `*` operator must be a pointer".to_string(),
                labels: vec![(span, "expression is not a pointer".into())],
                notes: vec![],
            },
            &Error::NotReference(what, span) => ParseError::Single {
                message: format!("{what} must be a reference"),
                labels: vec![(span, "expression is not a reference".into())],
                notes: vec![],
            },
            &Error::InvalidAssignment { span, ref ty } => {
                let (extra_label, notes) = match ty {
                    InvalidAssignmentType::Swizzle => (
                        None,
                        vec![
                            "WGSL does not support assignments to swizzles".into(),
                            "consider assigning each component individually".into(),
                        ],
                    ),
                    InvalidAssignmentType::ImmutableBinding(binding_span) => (
                        Some((*binding_span, "this is an immutable binding".into())),
                        vec![format!(
                            "consider declaring `{}` with `var` instead of `let`",
                            &source[*binding_span]
                        )],
                    ),
                    InvalidAssignmentType::Other => (None, vec![]),
                };

                ParseError::Single {
                    message: "invalid left-hand side of assignment".into(),
                    labels: core::iter::once((span, "cannot assign to this expression".into()))
                        .chain(extra_label)
                        .collect(),
                    notes,
                }
            },
            &Error::ReservedKeyword(name_span) => ParseError::Single {
                message: format!("name `{}` is a reserved keyword", &source[name_span]),
                labels: vec![(
                    name_span,
                    format!("definition of `{}`", &source[name_span]).into(),
                )],
                notes: vec![],
            },
            &Error::Redefinition { previous, current } => ParseError::Single {
                message: format!("redefinition of `{}`", &source[current]),
                labels: vec![
                    (
                        current,
                        format!("redefinition of `{}`", &source[current]).into(),
                    ),
                    (
                        previous,
                        format!("previous definition of `{}`", &source[previous]).into(),
                    ),
                ],
                notes: vec![],
            },
            &Error::RecursiveDeclaration { ident, usage } => ParseError::Single {
                message: format!("declaration of `{}` is recursive", &source[ident]),
                labels: vec![(ident, "".into()), (usage, "uses itself here".into())],
                notes: vec![],
            },
            &Error::CyclicDeclaration { ident, ref path } => ParseError::Single {
                message: format!("declaration of `{}` is cyclic", &source[ident]),
                labels: path
                    .iter()
                    .enumerate()
                    .flat_map(|(i, &(ident, usage))| {
                        [
                            (ident, "".into()),
                            (
                                usage,
                                if i == path.len() - 1 {
                                    "ending the cycle".into()
                                } else {
                                    format!("uses `{}`", &source[ident]).into()
                                },
                            ),
                        ]
                    })
                    .collect(),
                notes: vec![],
            },
            &Error::InvalidSwitchSelector { span } => ParseError::Single {
                message: "invalid `switch` selector".to_string(),
                labels: vec![(
                    span,
                    "`switch` selector must be a scalar integer"
                    .into(),
                )],
                notes: vec![],
            },
            &Error::InvalidSwitchCase { span } => ParseError::Single {
                message: "invalid `switch` case selector value".to_string(),
                labels: vec![(
                    span,
                    "`switch` case selector must be a scalar integer const expression"
                    .into(),
                )],
                notes: vec![],
            },
            &Error::SwitchCaseTypeMismatch { span } => ParseError::Single {
                message: "invalid `switch` case selector value".to_string(),
                labels: vec![(
                    span,
                    "`switch` case selector must have the same type as the `switch` selector expression"
                    .into(),
                )],
                notes: vec![],
            },
            &Error::CalledEntryPoint(span) => ParseError::Single {
                message: "entry point cannot be called".to_string(),
                labels: vec![(span, "entry point cannot be called".into())],
                notes: vec![],
            },
            &Error::WrongArgumentCount {
                span,
                ref expected,
                found,
            } => ParseError::Single {
                message: format!(
                    "wrong number of arguments: expected {}, found {}",
                    if expected.len() < 2 {
                        format!("{}", expected.start)
                    } else {
                        format!("{}..{}", expected.start, expected.end)
                    },
                    found
                ),
                labels: vec![(span, "wrong number of arguments".into())],
                notes: vec![],
            },
            &Error::TooManyArguments {
                ref function,
                call_span,
                arg_span,
                max_arguments,
            } => ParseError::Single {
                message: format!("too many arguments passed to `{function}`"),
                labels: vec![
                    (call_span, "".into()),
                    (arg_span, format!("unexpected argument #{}", max_arguments + 1).into())
                ],
                notes: vec![
                    format!("The `{function}` function accepts at most {max_arguments} argument(s)")
                ],
            },
            &Error::WrongArgumentType {
                ref function,
                call_span,
                arg_span,
                arg_index,
                ref arg_ty,
                ref allowed,
            } => {
                let message = format!(
                    "wrong type passed as argument #{} to `{function}`",
                    arg_index + 1,
                );
                let labels = vec![
                    (call_span, "".into()),
                    (arg_span, format!("argument #{} has type `{arg_ty}`", arg_index + 1).into())
                ];

                let mut notes = vec![];
                notes.push(format!("`{function}` accepts the following types for argument #{}:", arg_index + 1));
                notes.extend(allowed.iter().map(|ty| format!("allowed type: {ty}")));

                ParseError::Single { message, labels, notes }
            },
            &Error::InconsistentArgumentType {
                ref function,
                call_span,
                arg_span,
                arg_index,
                ref arg_ty,
                inconsistent_span,
                inconsistent_index,
                ref inconsistent_ty,
                ref allowed
            } => {
                let message = format!(
                    "inconsistent type passed as argument #{} to `{function}`",
                    arg_index + 1,
                );
                let labels = vec![
                    (call_span, "".into()),
                    (arg_span, format!("argument #{} has type {arg_ty}", arg_index + 1).into()),
                    (inconsistent_span, format!(
                        "this argument has type {inconsistent_ty}, which constrains subsequent arguments"
                    ).into()),
                ];
                let mut notes = vec![
                    format!("Because argument #{} has type {inconsistent_ty}, only the following types", inconsistent_index + 1),
                    format!("(or types that automatically convert to them) are accepted for argument #{}:", arg_index + 1),
                ];
                notes.extend(allowed.iter().map(|ty| format!("allowed type: {ty}")));

                ParseError::Single { message, labels, notes }
            }
            &Error::FunctionReturnsVoid(span) => ParseError::Single {
                message: "function does not return any value".to_string(),
                labels: vec![(span, "".into())],
                notes: vec![
                    "perhaps you meant to call the function in a separate statement?".into(),
                ],
            },
            &Error::FunctionMustUseUnused(call) => ParseError::Single {
                message: "unused return value from function annotated with @must_use".into(),
                labels: vec![(call, "".into())],
                notes: vec![
                    format!(
                        "function '{}' is declared with `@must_use` attribute",
                        &source[call],
                    ),
                    "use a phony assignment or declare a value using the function call as the initializer".into(),
                ],
            },
            &Error::FunctionMustUseReturnsVoid(attr, signature) => ParseError::Single {
                message: "function annotated with @must_use but does not return any value".into(),
                labels: vec![
                    (attr, "".into()),
                    (signature, "".into()),
                ],
                notes: vec![
                    "declare a return type or remove the attribute".into(),
                ],
            },
            &Error::InvalidWorkGroupUniformLoad(span) => ParseError::Single {
                message: "incorrect type passed to workgroupUniformLoad".into(),
                labels: vec![(span, "".into())],
                notes: vec!["passed type must be a workgroup pointer".into()],
            },
            &Error::Internal(ref message) => ParseError::Single {
                message: "internal WGSL front end error".to_string(),
                labels: vec![],
                notes: vec![message.to_string()],
            },
            &Error::ExpectedConstExprConcreteIntegerScalar(span) => ParseError::Single {
                message: concat!(
                    "must be a const-expression that ",
                    "resolves to a concrete integer scalar (`u32` or `i32`)"
                )
                .to_string(),
                labels: vec![(span, "must resolve to `u32` or `i32`".into())],
                notes: vec![],
            },
            &Error::ExpectedNonNegative(span) => ParseError::Single {
                message: "must be non-negative (>= 0)".to_string(),
                labels: vec![(span, "must be non-negative".into())],
                notes: vec![],
            },
            &Error::ExpectedPositiveArrayLength(span) => ParseError::Single {
                message: "array element count must be positive (> 0)".to_string(),
                labels: vec![(span, "must be positive".into())],
                notes: vec![],
            },
            &Error::ConstantEvaluatorError(ref e, span) => ParseError::Single {
                message: e.to_string(),
                labels: vec![(span, "see msg".into())],
                notes: vec![],
            },
            &Error::MissingWorkgroupSize(span) => ParseError::Single {
                message: "workgroup size is missing on compute shader entry point".to_string(),
                labels: vec![(
                    span,
                    "must be paired with a `@workgroup_size` attribute".into(),
                )],
                notes: vec![],
            },
            &Error::AutoConversion(ref error) => {
                // destructuring ensures all fields are handled
                let AutoConversionError {
                    dest_span,
                    ref dest_type,
                    source_span,
                    ref source_type,
                } = **error;
                ParseError::Single {
                    message: format!(
                        "automatic conversions cannot convert `{}` to `{}`",
                        source_type, dest_type
                    ),
                    labels: vec![
                        (
                            dest_span,
                            format!("a value of type {dest_type} is required here").into(),
                        ),
                        (
                            source_span,
                            format!("this expression has type {source_type}").into(),
                        ),
                    ],
                    notes: vec![],
                }
            }
            &Error::AutoConversionLeafScalar(ref error) => {
                let AutoConversionLeafScalarError {
                    dest_span,
                    ref dest_scalar,
                    source_span,
                    ref source_type,
                } = **error;
                ParseError::Single {
                    message: format!(
                        "automatic conversions cannot convert elements of `{}` to `{}`",
                        source_type, dest_scalar
                    ),
                    labels: vec![
                        (
                            dest_span,
                            format!(
                                "a value with elements of type {} is required here",
                                dest_scalar
                            )
                            .into(),
                        ),
                        (
                            source_span,
                            format!("this expression has type {source_type}").into(),
                        ),
                    ],
                    notes: vec![],
                }
            }
            &Error::ConcretizationFailed(ref error) => {
                let ConcretizationFailedError {
                    expr_span,
                    ref expr_type,
                    ref scalar,
                    ref inner,
                } = **error;
                ParseError::Single {
                    message: format!("failed to convert expression to a concrete type: {inner}"),
                    labels: vec![(
                        expr_span,
                        format!("this expression has type {expr_type}").into(),
                    )],
                    notes: vec![format!(
                        "the expression should have been converted to have {} scalar type",
                        scalar
                    )],
                }
            }
            &Error::ExceededLimitForNestedBraces { span, limit } => ParseError::Single {
                message: "brace nesting limit reached".into(),
                labels: vec![(span, "limit reached at this brace".into())],
                notes: vec![format!("nesting limit is currently set to {limit}")],
            },
            &Error::PipelineConstantIDValue(span) => ParseError::Single {
                message: "pipeline constant ID must be between 0 and 65535 inclusive".to_string(),
                labels: vec![(span, "must be between 0 and 65535 inclusive".into())],
                notes: vec![],
            },
            &Error::NotBool(span) => ParseError::Single {
                message: "must be a const-expression that resolves to a `bool`".to_string(),
                labels: vec![(span, "must resolve to `bool`".into())],
                notes: vec![],
            },
            &Error::ConstAssertFailed(span) => ParseError::Single {
                message: "`const_assert` failure".to_string(),
                labels: vec![(span, "evaluates to `false`".into())],
                notes: vec![],
            },
            &Error::DirectiveAfterFirstGlobalDecl { directive_span } => ParseError::Single {
                message: "expected global declaration, but found a global directive".into(),
                labels: vec![(
                    directive_span,
                    "written after first global declaration".into(),
                )],
                notes: vec![concat!(
                    "global directives are only allowed before global declarations; ",
                    "maybe hoist this closer to the top of the shader module?"
                )
                .into()],
            },
            &Error::EnableExtensionNotYetImplemented { kind, span } => ParseError::Single {
                message: format!(
                    "the `{}` enable-extension is not yet supported",
                    EnableExtension::Unimplemented(kind).to_ident()
                ),
                labels: vec![(
                    span,
                    concat!(
                        "this enable-extension specifies standard functionality ",
                        "which is not yet implemented in Naga"
                    )
                    .into(),
                )],
                notes: vec![format!(
                    concat!(
                        "Let Naga maintainers know that you ran into this at ",
                        "<https://github.com/gfx-rs/wgpu/issues/{}>, ",
                        "so they can prioritize it!"
                    ),
                    kind.tracking_issue_num()
                )],
            },
            &Error::EnableExtensionNotEnabled { kind, span } => ParseError::Single {
                message: format!("the `{}` enable extension is not enabled", kind.to_ident()),
                labels: vec![(
                    span,
                    format!(
                        concat!(
                            "the `{}` \"Enable Extension\" is needed for this functionality, ",
                            "but it is not currently enabled."
                        ),
                        kind.to_ident()
                    )
                    .into(),
                )],
                notes: if let EnableExtension::Unimplemented(kind) = kind {
                    vec![format!(
                        concat!(
                            "This \"Enable Extension\" is not yet implemented. ",
                            "Let Naga maintainers know that you ran into this at ",
                            "<https://github.com/gfx-rs/wgpu/issues/{}>, ",
                            "so they can prioritize it!"
                        ),
                        kind.tracking_issue_num()
                    )]
                } else {
                    vec![
                        format!(
                            "You can enable this extension by adding `enable {};` at the top of the shader, before any other items.",
                            kind.to_ident()
                        ),
                    ]
                },
            },
            &Error::LanguageExtensionNotYetImplemented { kind, span } => ParseError::Single {
                message: format!(
                    "the `{}` language extension is not yet supported",
                    LanguageExtension::Unimplemented(kind).to_ident()
                ),
                labels: vec![(span, "".into())],
                notes: vec![format!(
                    concat!(
                        "Let Naga maintainers know that you ran into this at ",
                        "<https://github.com/gfx-rs/wgpu/issues/{}>, ",
                        "so they can prioritize it!"
                    ),
                    kind.tracking_issue_num()
                )],
            },
            &Error::DiagnosticInvalidSeverity {
                severity_control_name_span,
            } => ParseError::Single {
                message: "invalid `diagnostic(…)` severity".into(),
                labels: vec![(
                    severity_control_name_span,
                    "not a valid severity level".into(),
                )],
                notes: vec![concat!(
                    "See available severities at ",
                    "<https://www.w3.org/TR/WGSL/#diagnostic-severity>."
                )
                .into()],
            },
            &Error::DiagnosticDuplicateTriggeringRule(ConflictingDiagnosticRuleError {
                triggering_rule_spans,
            }) => {
                let [first_span, second_span] = triggering_rule_spans;
                ParseError::Single {
                    message: "found conflicting `diagnostic(…)` rule(s)".into(),
                    labels: vec![
                        (first_span, "first rule".into()),
                        (second_span, "second rule".into()),
                    ],
                    notes: vec![
                        concat!(
                            "Multiple `diagnostic(…)` rules with the same rule name ",
                            "conflict unless they are directives and the severity is the same.",
                        )
                        .into(),
                        "You should delete the rule you don't want.".into(),
                    ],
                }
            },
            &Error::DiagnosticAttributeNotYetImplementedAtParseSite {
                site_name_plural,
                ref spans,
            } => {
                let mut spans_iter = spans.iter().cloned();
                let first = spans_iter
                    .next()
                    .map(|span| {
                        (
                            span,
                            format!("can't use this on {site_name_plural} (yet)").into(),
                        )
                    })
                    .expect("internal error: diag. attr. rejection on empty map");
                let labels = core::iter::once(first)
                    .chain(spans_iter.map(|span| (span, "".into())))
                    .collect();
                ParseError::Single {
                    message: "`@diagnostic(…)` attribute(s) not yet implemented".into(),
                    labels,
                    notes: vec![format!(concat!(
                        "Let Naga maintainers know that you ran into this at ",
                        "<https://github.com/gfx-rs/wgpu/issues/5320>, ",
                        "so they can prioritize it!"
                    ))],
                }
            },
            &Error::DiagnosticAttributeNotSupported { on_what, ref spans } => {
                // In this case the user may have intended to create a global diagnostic filter directive,
                // so display a note to them suggesting the correct syntax.
                let intended_diagnostic_directive = match on_what {
                    DiagnosticAttributeNotSupportedPosition::SemicolonInModulePosition => true,
                    DiagnosticAttributeNotSupportedPosition::Other { .. } => false,
                };
                let on_what_plural = match on_what {
                    DiagnosticAttributeNotSupportedPosition::SemicolonInModulePosition => {
                        "semicolons"
                    }
                    DiagnosticAttributeNotSupportedPosition::Other { display_plural } => {
                        display_plural
                    }
                };
                ParseError::Single {
                    message: format!(
                        "`@diagnostic(…)` attribute(s) on {on_what_plural} are not supported",
                    ),
                    labels: spans
                        .iter()
                        .cloned()
                        .map(|span| (span, "".into()))
                        .collect(),
                    notes: vec![
                        concat!(
                            "`@diagnostic(…)` attributes are only permitted on `fn`s, ",
                            "some statements, and `switch`/`loop` bodies."
                        )
                        .into(),
                        {
                            if intended_diagnostic_directive {
                                concat!(
                                    "If you meant to declare a diagnostic filter that ",
                                    "applies to the entire module, move this line to ",
                                    "the top of the file and remove the `@` symbol."
                                )
                                .into()
                            } else {
                                concat!(
                                    "These attributes are well-formed, ",
                                    "you likely just need to move them."
                                )
                                .into()
                            }
                        },
                    ],
                }
            },
            &Error::SelectUnexpectedArgumentType { arg_span, ref arg_type } => ParseError::Single {
                message: "unexpected argument type for `select` call".into(),
                labels: vec![(arg_span, format!("this value of type {arg_type}").into())],
                notes: vec!["expected a scalar or a `vecN` of scalars".into()],
            },
            &Error::SelectRejectAndAcceptHaveNoCommonType {
                reject_span,
                ref reject_type,
                accept_span,
                ref accept_type,
            } => ParseError::Single {
                message: "type mismatch for reject and accept values in `select` call".into(),
                labels: vec![
                    (reject_span, format!("reject value of type {reject_type}").into()),
                    (accept_span, format!("accept value of type {accept_type}").into()),
                ],
                notes: vec![],
            },
            // Fallback arm for non-exhaustive match
            _ => ParseError::Single {
                message: "Unknown error variant".to_string(),
                labels: vec![],
                notes: vec![],
            },
        }
    }
}
