//! Synthetic input for `frames` surfaces and automated remote windows.
//!
//! `child` and `window` surfaces receive a person's OS input directly; a
//! `frames` surface is composited by the host, so the host forwards input
//! explicitly via [`WebView::input`](crate::WebView::input), as automation
//! does for a remote engine's window. Coordinates are in CSS pixels of the
//! page viewport (device pixels divided by the frame scale).

use omp_core::Str;

/// A mouse button, named as the automation protocols expect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display, strum::IntoStaticStr)]
#[strum(serialize_all = "lowercase")]
pub enum MouseButton {
	/// Primary button.
	Left,
	/// Wheel button.
	Middle,
	/// Secondary button.
	Right,
}

/// Held modifier keys accompanying an input event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Modifiers {
	/// Alt / Option.
	pub alt:   bool,
	/// Control.
	pub ctrl:  bool,
	/// Meta / Command / Windows.
	pub meta:  bool,
	/// Shift.
	pub shift: bool,
}

impl Modifiers {
	/// No modifiers held.
	pub const NONE: Self = Self { alt: false, ctrl: false, meta: false, shift: false };
}

/// A key identity for [`Input::KeyDown`]/[`Input::KeyUp`].
///
/// Printable input should prefer [`Input::Text`]; `Key::Char` exists for
/// shortcuts (e.g. ctrl+c) where the physical key matters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Key {
	/// A printable character key.
	Char(char),
	/// Return / Enter.
	Enter,
	/// Tab.
	Tab,
	/// Backspace.
	Backspace,
	/// Forward delete.
	Delete,
	/// Escape.
	Escape,
	/// Up arrow.
	ArrowUp,
	/// Down arrow.
	ArrowDown,
	/// Left arrow.
	ArrowLeft,
	/// Right arrow.
	ArrowRight,
	/// Home.
	Home,
	/// End.
	End,
	/// Page up.
	PageUp,
	/// Page down.
	PageDown,
	/// Function key `F1`..=`F12`.
	F(u8),
}

/// One synthetic input event, forwarded to the page by the engine driver.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Input {
	/// Pointer moved to `(x, y)`.
	MouseMove {
		/// X in CSS pixels.
		x: f64,
		/// Y in CSS pixels.
		y: f64,
	},
	/// Button pressed at `(x, y)`.
	MouseDown {
		/// Button pressed.
		button: MouseButton,
		/// X in CSS pixels.
		x:      f64,
		/// Y in CSS pixels.
		y:      f64,
		/// Click count (1 = single, 2 = double).
		clicks: u8,
	},
	/// Button released at `(x, y)`.
	MouseUp {
		/// Button released.
		button: MouseButton,
		/// X in CSS pixels.
		x:      f64,
		/// Y in CSS pixels.
		y:      f64,
	},
	/// Wheel scrolled at `(x, y)` by `(dx, dy)` CSS pixels.
	Scroll {
		/// X in CSS pixels.
		x:  f64,
		/// Y in CSS pixels.
		y:  f64,
		/// Horizontal delta.
		dx: f64,
		/// Vertical delta.
		dy: f64,
	},
	/// Key pressed.
	KeyDown {
		/// Key identity.
		key:       Key,
		/// Held modifiers.
		modifiers: Modifiers,
	},
	/// Key released.
	KeyUp {
		/// Key identity.
		key:       Key,
		/// Held modifiers.
		modifiers: Modifiers,
	},
	/// Insert text as typed (IME-style), without synthesizing raw key events.
	Text(Str),
}
