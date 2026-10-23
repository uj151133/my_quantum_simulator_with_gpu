#include "Core.hpp"

const unordered_set<string> cancerSTAGE4 = {"H", "V", "Vdg", "VDG"};
const unordered_set<string> cancerSTAGE3 = {"Rx", "RX", "Ry", "RY"};

bool Core::isCancer() const {
    return (cancerSTAGE4.contains(this->tag) || (cancerSTAGE3.contains(this->tag)) && !(mathUtils::isMultiplePI(this->theta)));
}

string Core::upper(string s){
    for (auto& c : s) c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
    return s;
}

bool Core::isDiagTag(const string& tagU){
    const string t = upper(tagU);
    if (t=="RZ"||t=="U1"||t=="P"||t=="S"||t=="T"||t=="Z") return true;
    if (t=="CZ"||t=="CP"||t=="CRZ"||t=="RZZ") return true;
    return false;
}

string Core::tagToShape(const string& tagU){
    const string t = upper(tagU);
    if (t == "FUSED") return kShapeFused;
    if (isDiagTag(t)) return kShapeDiag;
    if (t == "X" || t == "SWAP") return kShapePerm;
    if (t == "Y") return kShapeAnti;
    return kShapeGeneral;
}

bool Core::isSymmetric2QTag(const string& tagU){
    const string t = upper(tagU);
    return (t=="CZ" || t=="RZZ" || t=="SWAP");
}

pair<int,int> Core::orderedPair() const{
    return (this->qubits.size()==2) ? pair<int,int>{this->qubits[0], this->qubits[1]} : pair<int,int>{-1,-1};
}

pair<int,int> Core::unorderedKey() const{
    if (this->qubits.size()!=2) return {-1,-1};
    int a = this->qubits[0], b = this->qubits[1];
    return (a<b) ? pair<int,int>{a,b} : pair<int,int>{b,a};
}

void Core::normalize(){
    this->tag = upper(this->tag);

    {
        unordered_set<int> seen;
        vector<int> uniq; uniq.reserve(this->qubits.size());
        for(int q : this->qubits){
            if(seen.insert(q).second) uniq.push_back(q);
        }
        this->qubits.swap(uniq);
    }

    this->shape = tagToShape(this->tag);
}