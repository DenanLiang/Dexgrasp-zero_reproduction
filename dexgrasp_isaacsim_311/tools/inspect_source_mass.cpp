#include <raisim/World.hpp>
#include <iostream>
int main(int argc,char**argv){raisim::World w;for(int i=1;i<argc;i++){auto*a=w.addArticulatedSystem(argv[i]);std::cout<<argv[i]<<" MASS "<<a->getTotalMass()<<" LIMITS "<<a->getActuationUpperLimits().e().transpose()<<"\n";}}
