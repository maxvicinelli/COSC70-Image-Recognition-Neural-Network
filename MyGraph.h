#ifndef __MyGraph_h__
#define __MyGraph_h__
#include <utility>
#include "MyMatrix.h"
#include "MySparseMatrix.h"

class Graph
{
public:
	vector<double> node_values;				////values on nodes
	vector<double> edge_values;				////values on edges
	vector<pair<int,int> > edges;		////edges connecting nodes

	void Add_Node(const double& value)
	{
		node_values.push_back(value);
	}

	void Add_Edge(const int i,const int j,const double value=1.)
	{
		edges.push_back(pair<int,int>(i,j));
		edge_values.push_back(value);
	}

	////display graph in terminal
	friend ostream & operator << (ostream &out,const Graph &graph)
	{
		cout<<"graph node values: "<<graph.node_values.size()<<endl;
		for(int i=0;i<(int)graph.node_values.size();i++){
			cout<<"["<<i<<", "<<graph.node_values[i]<<"] ";
		}
		cout<<endl;

		cout<<"graph edge values: "<<graph.edge_values.size()<<endl;
		for(int i=0;i<(int)graph.edge_values.size();i++){
			cout<<"["<<i<<", "<<graph.edge_values[i]<<"] ";
		}
		cout<<endl;

		cout<<"graph edges: "<<graph.edges.size()<<endl;
		for(int i=0;i<(int)graph.edges.size();i++){
			cout<<"["<<graph.edges[i].first<<", "<<graph.edges[i].second<<"] ";
		}
		cout<<endl;

		return out;
	}

	//////////////////////////////////////////////////////////////////////////
	////Your homework starts

	////HW4 Task 0: build incidence matrix
	void Incidence_Matrix(/*result*/Matrix& inc_m)
	{
		/*Your implementation starts*/
		/*Your implementation ends*/
	}

	////HW4 Task 1: build adjancency matrix
	void Adjacency_Matrix(/*result*/Matrix& adj_m)
	{
		/*Your implementation starts*/
		/*Your implementation ends*/
	}

	////HW4 Task 3: build the negative Laplacian matrix
	void Laplacian_Matrix(/*result*/Matrix& lap_m)
	{
		/*Your implementation starts*/
		/*Your implementation ends*/
	}

	////HW4 Task 4: calculate the Dirichlet energy
	double Dirichlet_Energy(const vector<double>& v)
	{
		double de=(double)0;
		/*Your implementation starts*/
		/*Your implementation ends*/

		return de;
	}

	////HW4 Task 5: smooth the node values on the graph by iteratively applying the Laplacian matrix
	void Smooth_Node_Values(const double dt,const int iter_num)
	{
		////copy node_values to local variables
		int m=(int)node_values.size();
		Matrix v(m,1);
		for(int i=0;i<m;i++){
			v(i,0)=node_values[i];
		}
		Matrix v2=v;

		////smooth v
		/*Your implementation starts*/
		/*Your implementation ends*/

		////copy local variables to node_values
		for(int i=0;i<m;i++){
			node_values[i]=v(i,0);
		}
	}
};

#endif