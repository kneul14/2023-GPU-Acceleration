#pragma once
#include <glm/glm.hpp>
#include "GameObject.h"
#include "Particle.h"
#include <vector>

class ForceGenerator
{
private:
	glm::vec3 gravity = glm::vec3(0, -9.8, 0);
	glm::vec3 drag = glm::vec3(2.5, 5, 0);


protected:
	std::vector<GameObject*> RegObj;
public:

	ForceGenerator();
	~ForceGenerator();
	virtual void Update(float deltaTime);
	void AddObjectsToFunc(GameObject*);
};

