#ifndef TDLS_CORE_VERSION_HPP
#define TDLS_CORE_VERSION_HPP



/// \file
/// \brief TDLS version macros.
/// \author Tristan Chenaille

/// \def TDLS_VERSION_MAJOR
/// \brief Major version component.
#define TDLS_VERSION_MAJOR 0

/// \def TDLS_VERSION_MINOR
/// \brief Minor version component.
#define TDLS_VERSION_MINOR 1

/// \def TDLS_VERSION_PATCH
/// \brief Patch version component.
#define TDLS_VERSION_PATCH 0

/// \def TDLS_VERSION_STRING
/// \brief Version as a "major.minor.patch" string literal.
#define TDLS_VERSION_STRING "0.1.0"

/// \def TDLS_VERSION
/// \brief Single comparable value: major * 10000 + minor * 100 + patch.
#define TDLS_VERSION (TDLS_VERSION_MAJOR * 10000 + TDLS_VERSION_MINOR * 100 + TDLS_VERSION_PATCH)



#endif // TDLS_CORE_VERSION_HPP
