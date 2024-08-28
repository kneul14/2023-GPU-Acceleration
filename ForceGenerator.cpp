#include "ForceGenerator.h"

ForceGenerator::ForceGenerator()
{

}


ForceGenerator::~ForceGenerator()
{

}
//regiobj id.x for threads
void ForceGenerator::Update(float deltaTime)
{
	for (int i = 0; i < RegObj.size(); i++)
	{
		RegObj[i]->GetParticle()->AddForce(gravity);
		RegObj[i]->GetParticle()->AddForce(drag);
	}
}

void ForceGenerator::AddObjectsToFunc(GameObject* obj)
{
	RegObj.push_back(obj); // adds objs to the list/array automatically:)
}
