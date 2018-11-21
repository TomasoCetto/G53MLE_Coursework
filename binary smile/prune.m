use node to traverse through the tree 
use tree to keep track of the origin root node of the tree
use parent to keep track of the parent of the node, so as to manipulate the new tree
inputs, targets oldf1 are all from testing data of 1 fold cv

function newTree = prune(tree, node, parent, inputs, targets, oldf1)
	newTree = tree
	newNode = node

if isempty(node)
	% newTree = tree
	return

else 
	replace it with leaf with class = major(tree)
	newNode.kids = [];

	if new f1 > old f1
		newTree = tree
		return;
	else
		prune(tree, node.kids{1,1}, node, inputs, targets, oldf1);
