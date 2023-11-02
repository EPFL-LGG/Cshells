# - Find Knitro
#  Searches for includes/libraries using environment variable $KNITRO_PATH or $KNITRO_DIR
#  Knitro_INCLUDE_DIRS - where to find knitro.h and (separately) the c++ interface
#  Knitro_LIBRARIES    - List of libraries needed to use knitro.
#  Knitro_FOUND        - True if knitro found.


IF (Knitro_INCLUDE_DIRS)
  # Already in cache, be silent
  SET (knitro_FIND_QUIETLY TRUE)
ENDIF (Knitro_INCLUDE_DIRS)


FIND_PATH(Knitro_INCLUDE_DIR knitro.h
	HINTS
        $ENV{KNITRO_PATH}/include
        $ENV{KNITRO_DIR}/include
        $ENV{KNITRODIR}/include         # Recommended by Artelys
        /usr/local/opt/knitro/include   # If manual installation and no env var
        /opt/knitro/include             # If manual installation and no env var
)

FIND_LIBRARY (Knitro_LIBRARY NAMES knitro knitro1240 knitro1222 knitro1031
	HINTS
        $ENV{KNITRO_PATH}/lib
        $ENV{KNITRO_DIR}/lib
        $ENV{KNITRODIR}/lib
        /usr/local/opt/knitro/lib
        /opt/knitro/lib
)

# handle the QUIETLY and REQUIRED arguments and set Knitro_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (Knitro DEFAULT_MSG
  Knitro_LIBRARY
  Knitro_INCLUDE_DIR)

SET (Knitro_CPP_Examples "${Knitro_INCLUDE_DIR}/../examples/C++")

IF(Knitro_FOUND)
    SET (Knitro_LIBRARIES ${Knitro_LIBRARY})
    SET (Knitro_INCLUDE_DIRS "${Knitro_INCLUDE_DIR}" "${Knitro_INCLUDE_DIR}/../examples/C++/include")
    MESSAGE("   Knitro lib path    : ${Knitro_LIBRARIES}")
    MESSAGE("   Knitro include path: ${Knitro_INCLUDE_DIRS}")

    IF (EXISTS "${Knitro_CPP_Examples}/src")
        file(GLOB INTERFACE_SOURCES ${Knitro_CPP_Examples}/src/*.cpp)
        add_library(knitrocpp SHARED ${INTERFACE_SOURCES})
        target_link_libraries(knitrocpp PUBLIC ${Knitro_LIBRARIES})
        target_include_directories(knitrocpp PUBLIC SYSTEM ${Knitro_INCLUDE_DIRS})
        set_property(TARGET knitrocpp PROPERTY CXX_STANDARD 11)
        add_library(knitro::knitro ALIAS knitrocpp)
        target_compile_definitions(knitrocpp PUBLIC -DHAS_KNITRO)
    ELSE()
        add_library(knitrocpp INTERFACE ${INTERFACE_SOURCES})
        target_link_libraries(knitrocpp INTERFACE ${Knitro_LIBRARIES})
        target_include_directories(knitrocpp INTERFACE SYSTEM ${Knitro_INCLUDE_DIRS})
        target_compile_definitions(knitrocpp INTERFACE -DHAS_KNITRO -DKNITRO_LEGACY)
        add_library(knitro::knitro ALIAS knitrocpp)
    ENDIF()
ELSE()
ENDIF (Knitro_FOUND)



MARK_AS_ADVANCED (Knitro_LIBRARY Knitro_INCLUDE_DIR Knitro_INCLUDE_DIRS Knitro_LIBRARIES)
