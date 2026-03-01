# Findglfw3.cmake
#
# Finds the GLFW3 library
#
# This will define the following variables
#
#   glfw3_FOUND
#   glfw3_INCLUDE_DIRS
#   glfw3_LIBRARIES
#
# and the following imported targets
#
#   glfw::glfw
#

# Try to find the package using pkg-config first
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PC_glfw3 QUIET glfw3)
endif()

# 1. Find the include directory
find_path(glfw3_INCLUDE_DIR
  NAMES GLFW/glfw3.h
  PATHS
    ${PC_glfw3_INCLUDE_DIRS}
    /usr/include
    /usr/local/include
    $ENV{VULKAN_SDK}/include
    ${ANDROID_NDK}/sources/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../external
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/external
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../external
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../include
  PATH_SUFFIXES glfw
)

# 2. Find the library
find_library(glfw3_LIBRARY
  NAMES glfw glfw3
  PATHS
    ${PC_glfw3_LIBRARY_DIRS}
    /usr/lib
    /usr/local/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../external/lib
)

# If the include directory or library wasn't found, use FetchContent
if(NOT glfw3_INCLUDE_DIR OR NOT glfw3_LIBRARY)
  include(FetchContent)

  message(STATUS "GLFW3 not found, fetching from GitHub...")
  FetchContent_Declare(
    glfw3
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
  )

  # Pre-configure GLFW options to disable building extras
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

  # --- FIX FOR WARNING ---
  # We removed the manual FetchContent_Populate check.
  # FetchContent_MakeAvailable handles checking, populating, and adding the subdirectory
  # in a single step compliant with CMP0169.
  FetchContent_MakeAvailable(glfw3)

  # Get the include directory from the target created by GLFW
  if(TARGET glfw)
    get_target_property(glfw3_INCLUDE_DIR glfw INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT glfw3_INCLUDE_DIR)
      set(glfw3_INCLUDE_DIR ${glfw3_SOURCE_DIR}/include)
    endif()
  else()
    # Fallback if target wasn't created for some reason
    FetchContent_GetProperties(glfw3)
    set(glfw3_INCLUDE_DIR ${glfw3_SOURCE_DIR}/include)
  endif()
endif()

# Set the variables
include(FindPackageHandleStandardArgs)

if(TARGET glfw)
    # If we built it ourselves, we consider it found
    set(glfw3_FOUND TRUE)
else()
    # If we are looking for system files, verify them
    find_package_handle_standard_args(glfw3 
      REQUIRED_VARS glfw3_INCLUDE_DIR glfw3_LIBRARY
    )
endif()

# --- TARGET CREATION ---

if(glfw3_FOUND AND NOT TARGET glfw)
  # CASE 1: Found on System (binary exists)
  set(glfw3_INCLUDE_DIRS ${glfw3_INCLUDE_DIR})
  set(glfw3_LIBRARIES ${glfw3_LIBRARY})

  if(NOT TARGET glfw::glfw)
    add_library(glfw::glfw UNKNOWN IMPORTED)
    set_target_properties(glfw::glfw PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${glfw3_INCLUDE_DIRS}"
      IMPORTED_LOCATION "${glfw3_LIBRARY}"
    )
  endif()

elseif(TARGET glfw)
  # CASE 2: Built via FetchContent (Source target exists)
  
  # Create the namespace alias
  if(NOT TARGET glfw::glfw)
    add_library(glfw::glfw ALIAS glfw)
  endif()

  set(glfw3_FOUND TRUE)
  set(GLFW3_FOUND TRUE)

  get_target_property(glfw3_INCLUDE_DIR glfw INTERFACE_INCLUDE_DIRECTORIES)
  if(glfw3_INCLUDE_DIR)
    set(glfw3_INCLUDE_DIRS ${glfw3_INCLUDE_DIR})
  else()
    FetchContent_GetProperties(glfw3)
    set(glfw3_INCLUDE_DIR ${glfw3_SOURCE_DIR}/include)
    set(glfw3_INCLUDE_DIRS ${glfw3_INCLUDE_DIR})
  endif()
  
endif()

mark_as_advanced(glfw3_INCLUDE_DIR glfw3_LIBRARY)