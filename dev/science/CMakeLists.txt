if(APPLE)
  set(CMAKE_C_FLAGS "-O2 -Wall -mmacosx-version-min=10.8 -Dunix")
else()
  if(MINGW)
    set(CMAKE_C_FLAGS "-O2 -Wall -DWIN32")
  else()
    set(CMAKE_C_FLAGS "-O2 -Wall -Dlinux -Dunix")
  endif()
endif()
#RWD only need this for APPLE?
link_directories (/usr/local/lib)
find_library(AAIOLIB NAMES libaaio.a)

link_directories(../cdp2k ../sfsys)

include_directories(../../include /usr/local/include)
#OUT
#add_executable(aminomap aminomap.c)
#target_link_libraries(aminomap ${EXTRA_LIBRARIES})
#my_install(aminomap)

add_executable(bounce bounce.c)
target_link_libraries(bounce  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(bounce)

add_executable(brownian brownian.c)
target_link_libraries(brownian  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(brownian)

add_executable(cascade cascade.c)
target_link_libraries(cascade  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(cascade)

add_executable(chirikov chirikov.c)
target_link_libraries(chirikov  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(chirikov)

#OUT
#add_executable(codonreader codonreader.c)
#target_link_libraries(codonreader ${EXTRA_LIBRARIES})
#my_install(codonreader)

add_executable(crumble crumble.c)
target_link_libraries(crumble  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(crumble)

add_executable(crystal crystal.c)
target_link_libraries(crystal  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(crystal)

add_executable(cubicspline cubicspline.c)
target_link_libraries(cubicspline cdp2k sfsys ${EXTRA_LIBRARIES})

add_executable(repeat distortt.c)
target_link_libraries(repeat  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(repeat)

add_executable(dvdwind dvdwind.c)
target_link_libraries(dvdwind  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(dvdwind)

# Seems old version of superaccu, errors prevent building.
# add_executable(example example.c)
# target_link_libraries(example  cdp2k sfsys ${EXTRA_LIBRARIES})
# my_install(example)

add_executable(flatten flatten.c)
target_link_libraries(flatten  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(flatten)

#OUT
#add_executable(hztomidi hztomidi.c)
#target_link_libraries(hztomidi ${EXTRA_LIBRARIES})
#my_install(hztomidi)

add_executable(impulse impulse.c)
target_link_libraries(impulse  cdp2k sfsys ${EXTRA_LIBRARIES})
my_install(impulse)


#OUT
#add_executable(integratetime integratetime.c)
#target_link_libraries(integratetime cdp2k sfsys ${EXTRA_LIBRARIES})
#my_install(integratetime)
#OUT
#add_executable(miditohz miditohz.c)
#target_link_libraries(miditohz ${EXTRA_LIBRARIES})
#my_install(miditohz)

add_executable(motor motor.c)
target_link_libraries(motor  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(motor)

add_executable(multiosc multiosc.c)
target_link_libraries(multiosc  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(multiosc)

add_executable(multisynth multisynth.c)
target_link_libraries(multisynth  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(multisynth)

add_executable(newscales newscales.c)
target_link_libraries(newscales  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(newscales)

add_executable(notchinvert notchinvert.c)
target_link_libraries(notchinvert  cdp2k sfsys ${EXTRA_LIBRARIES})
my_install(notchinvert)

add_executable(peakiso peakiso.c)
target_link_libraries(peakiso  cdp2k sfsys ${EXTRA_LIBRARIES})
my_install(peakiso)

#OUT
#add_executable(perms perms.c)
#target_link_libraries(perms ${EXTRA_LIBRARIES})
#my_install(perms)

add_executable(pulser pulser.c)
target_link_libraries(pulser  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(pulser)

add_executable(refocus refocus.c)
target_link_libraries(refocus  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(refocus)

add_executable(repeater repeater.c)
target_link_libraries(repeater  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(repeater)

#OUT
#add_executable(rhymorph rhymorph.c)
#target_link_libraries(rhymorph  cdp2k sfsys ${EXTRA_LIBRARIES})
#my_install(rhymorph)
#OUT
#add_executable(rhymorph2 rhymorph2.c)
#target_link_libraries(rhymorph2  cdp2k sfsys ${EXTRA_LIBRARIES})
#my_install(rhymorph2)

add_executable(scramble scramble.c)
target_link_libraries(scramble  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(scramble)

add_executable(smooth smooth.c)
target_link_libraries(smooth  cdp2k sfsys ${EXTRA_LIBRARIES})
my_install(smooth)

add_executable(sorter sorter.c)
target_link_libraries(sorter  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(sorter)

add_executable(specanal specanal.c)
target_link_libraries(specanal  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(specanal)

add_executable(specav specav.c)
target_link_libraries(specav  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(specav)

add_executable(specfnu specfnu.c)
target_link_libraries(specfnu  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(specfnu)

add_executable(specfold specfold.c)
target_link_libraries(specfold  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(specfold)

add_executable(spectstr spectstr.c)
target_link_libraries(spectstr  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(spectstr)

add_executable(spectune spectune.c)
target_link_libraries(spectune  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(spectune)

add_executable(spin spin.c)
target_link_libraries(spin  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(spin)

add_executable(splinter splinter.c)
target_link_libraries(splinter  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(splinter)

add_executable(strands strands.c)
target_link_libraries(strands  cdp2k sfsys ${AAIOLIB} ${EXTRA_LIBRARIES})

my_install(strands)

add_executable(stutter stutter.c)
target_link_libraries(stutter  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(stutter)

# error preventing build
 add_executable(synfilt synfilt.c)
 target_link_libraries(synfilt  cdp2k sfsys ${EXTRA_LIBRARIES})
 my_install(synfilt)

add_executable(synspline synspline.c)
target_link_libraries(synspline  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(synspline)

add_executable(synthesis synthesis.c)
target_link_libraries(synthesis  cdp2k sfsys ${AAIOLIB} ${EXTRA_LIBRARIES})

my_install(synthesis)

add_executable(tesselate tesselate.c)
target_link_libraries(tesselate  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(tesselate)

add_executable(ts timeseries.c)
target_link_libraries(ts  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(ts)

add_executable(tsconvert ts_convert.c)
target_link_libraries(tsconvert ${EXTRA_LIBRARIES})
my_install(tsconvert)

add_executable(tweet tweet.c)
target_link_libraries(tweet  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(tweet)

add_executable(unknot unknot.c)
target_link_libraries(unknot  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(unknot)

add_executable(verges verges.c)
target_link_libraries(verges  cdp2k sfsys ${EXTRA_LIBRARIES})

my_install(verges)

add_executable(waveform waveform.c)
target_link_libraries(waveform  cdp2k sfsys ${EXTRA_LIBRARIES})
my_install(waveform)

#OUT
#add_executable(wavmedian wavmedian.c)
#target_link_libraries(wavmedian  cdp2k sfsys ${EXTRA_LIBRARIES})
#my_install(wavmedian)