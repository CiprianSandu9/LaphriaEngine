# FindOpenAL.cmake
#
# Finds the OpenAL library or fetches openal-soft if not found.
#
# This will define the following variables:
#    OPENAL_FOUND
#    OPENAL_INCLUDE_DIR
#    OPENAL_LIBRARY
#
# and the following imported target:
#    OpenAL::OpenAL

set(_OPENAL_SYSTEM_FOUND FALSE)

# --- 1. Attempt to find on System ---

# MacOS usually has OpenAL as a framework.
# Linux usually has it via package manager.
# Windows usually needs it fetched.

if(APPLE)
  # On macOS, use the built-in framework
  find_library(OPENAL_LIBRARY OpenAL)
  find_path(OPENAL_INCLUDE_DIR OpenAL/al.h)
  
  if(OPENAL_LIBRARY AND OPENAL_INCLUDE_DIR)
    set(_OPENAL_SYSTEM_FOUND TRUE)
  endif()

elseif(UNIX)
  # On Linux, try PkgConfig first
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_OPENAL QUIET openal)
  endif()

  find_path(OPENAL_INCLUDE_DIR al.h
    HINTS
      ${PC_OPENAL_INCLUDEDIR}
      ${PC_OPENAL_INCLUDE_DIRS}
    PATH_SUFFIXES AL OpenAL include/AL include/OpenAL
  )

  find_library(OPENAL_LIBRARY
    NAMES openal OpenAL
    HINTS
      ${PC_OPENAL_LIBDIR}
      ${PC_OPENAL_LIBRARY_DIRS}
  )

  if(OPENAL_LIBRARY AND OPENAL_INCLUDE_DIR)
    set(_OPENAL_SYSTEM_FOUND TRUE)
  endif()
  
else()
  # On Windows, we often prefer the fallback, but let's try to find it 
  # just in case the user installed the SDK manually.
  find_path(OPENAL_INCLUDE_DIR al.h
    PATH_SUFFIXES include/AL include/OpenAL
    HINTS $ENV{OPENAL_DIR}/include
  )

  find_library(OPENAL_LIBRARY
    NAMES OpenAL32 openal
    PATH_SUFFIXES lib libs x64
    HINTS $ENV{OPENAL_DIR}/libs/Win64
  )

  if(OPENAL_LIBRARY AND OPENAL_INCLUDE_DIR)
    set(_OPENAL_SYSTEM_FOUND TRUE)
  endif()
endif()

# --- 2. Define Targets or FetchContent ---

if(_OPENAL_SYSTEM_FOUND)
  set(OPENAL_FOUND TRUE)
  
  if(NOT TARGET OpenAL::OpenAL)
    add_library(OpenAL::OpenAL UNKNOWN IMPORTED)
    set_target_properties(OpenAL::OpenAL PROPERTIES
      IMPORTED_LOCATION "${OPENAL_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${OPENAL_INCLUDE_DIR}"
    )
  endif()
  
  message(STATUS "Found System OpenAL: ${OPENAL_LIBRARY}")

else()
  # --- Fallback: FetchContent (openal-soft) ---
  
  message(STATUS "OpenAL not found on system. Fetching openal-soft from GitHub...")

  include(FetchContent)

  FetchContent_Declare(
    openal
    GIT_REPOSITORY https://github.com/kcat/openal-soft.git
    GIT_TAG        1.24.3 # Pinning a stable version is recommended
  )

  # Configure OpenAL-Soft options to keep the build light
  set(ALSOFT_UTILS OFF CACHE BOOL "Build utilities" FORCE)
  set(ALSOFT_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
  set(ALSOFT_TESTS OFF CACHE BOOL "Build tests" FORCE)
  set(ALSOFT_INSTALL OFF CACHE BOOL "Install OpenAL" FORCE)
  
  # If you are strictly using this for a game, you might want to force the backend
  # set(ALSOFT_BACKEND_DSOUND ON CACHE BOOL "" FORCE) # Example for Windows

  FetchContent_MakeAvailable(openal)

  # OpenAL-Soft creates a target named "OpenAL". 
  # We create an alias "OpenAL::OpenAL" to standardize usage.
  if(TARGET OpenAL AND NOT TARGET OpenAL::OpenAL)
    add_library(OpenAL::OpenAL ALIAS OpenAL)
  endif()

  set(OPENAL_FOUND TRUE)
endif()

# --- 3. Final Verification ---

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenAL
  REQUIRED_VARS OPENAL_FOUND
)

# Hide variables from the GUI if we found them
if(_OPENAL_SYSTEM_FOUND)
  mark_as_advanced(OPENAL_INCLUDE_DIR OPENAL_LIBRARY)
endif()