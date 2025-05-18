document.addEventListener('DOMContentLoaded', () => {
    const questionsListDiv = document.getElementById('questions-list');
    const hintModal = document.getElementById('hint-modal');
    const hintQuestionTitle = document.getElementById('hint-question');
    const hintContentDiv = document.getElementById('hint-content');
    const closeBtn = document.querySelector('.close-btn');

    // Complete list of Fasal coding questions with hints and answers
    const fasalQuestions = [
        // ... (your existing question array remains the same)
         {
            question: "Two Sum",
            description: "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
            hint: "Use a hash map to store the difference between target and current element as you iterate through the array.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>We can use a hash map to store the elements we've seen so far along with their indices. For each element, we calculate the complement (target - current element) and check if it exists in the map.</p>

                <pre><code>function twoSum(nums, target) {
        const map = new Map();
        for (let i = 0; i < nums.length; i++) {
            const complement = target - nums[i];
            if (map.has(complement)) {
                return [map.get(complement), i];
            }
            map.set(nums[i], i);
        }
        return [];
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n) - We traverse the list once</p>
                <p><strong>Space Complexity:</strong> O(n) - We store elements in a hash map</p>
            `
        },
        {
            question: "Merge Intervals",
            description: "Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals.",
            hint: "First sort the intervals by their start time, then merge adjacent intervals if they overlap.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>First sort the intervals based on their start values. Then iterate through them, merging the current interval with the last merged interval if they overlap.</p>

                <pre><code>function merge(intervals) {
        if (intervals.length <= 1) return intervals;

        intervals.sort((a, b) => a[0] - b[0]);

        const merged = [intervals[0]];

        for (let i = 1; i < intervals.length; i++) {
            const last = merged[merged.length - 1];
            const current = intervals[i];

            if (current[0] <= last[1]) {
                last[1] = Math.max(last[1], current[1]);
            } else {
                merged.push(current);
            }
        }

        return merged;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n log n) - Due to sorting</p>
                <p><strong>Space Complexity:</strong> O(n) - For storing merged intervals</p>
            `
        },
        {
            question: "LRU Cache",
            description: "Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.",
            hint: "Use a combination of hash map and doubly linked list to achieve O(1) time for both get and put operations.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>We use a Map object which maintains insertion order to implement the LRU cache. For each access, we delete and reinsert to maintain recency.</p>

                <pre><code>class LRUCache {
        constructor(capacity) {
            this.capacity = capacity;
            this.cache = new Map();
        }

        get(key) {
            if (!this.cache.has(key)) return -1;
            const value = this.cache.get(key);
            this.cache.delete(key);
            this.cache.set(key, value);
            return value;
        }

        put(key, value) {
            if (this.cache.has(key)) {
                this.cache.delete(key);
            } else if (this.cache.size >= this.capacity) {
                const firstKey = this.cache.keys().next().value;
                this.cache.delete(firstKey);
            }
            this.cache.set(key, value);
        }
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(1) for both get and put</p>
                <p><strong>Space Complexity:</strong> O(capacity)</p>
            `
        },
        {
            question: "Maximum Subarray",
            description: "Given an integer array nums, find the contiguous subarray which has the largest sum.",
            hint: "Use Kadane's algorithm to track the maximum subarray sum ending at each position.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>Kadane's algorithm maintains a running sum and resets it when it becomes negative, while keeping track of the maximum sum encountered.</p>

                <pre><code>function maxSubArray(nums) {
        let maxSum = nums[0];
        let currentSum = nums[0];

        for (let i = 1; i < nums.length; i++) {
            currentSum = Math.max(nums[i], currentSum + nums[i]);
            maxSum = Math.max(maxSum, currentSum);
        }

        return maxSum;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n)</p>
                <p><strong>Space Complexity:</strong> O(1)</p>
            `
        },
        {
            question: "Best Time to Buy and Sell Stock",
            description: "Given an array prices where prices[i] is the price of a stock on day i, find the maximum profit.",
            hint: "Track the minimum price seen so far and calculate potential profit at each day.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>Iterate through prices while keeping track of the minimum price. Calculate potential profit at each step.</p>

                <pre><code>function maxProfit(prices) {
        let minPrice = Infinity;
        let maxProfit = 0;

        for (let price of prices) {
            if (price < minPrice) {
                minPrice = price;
            } else if (price - minPrice > maxProfit) {
                maxProfit = price - minPrice;
            }
        }

        return maxProfit;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n)</p>
                <p><strong>Space Complexity:</strong> O(1)</p>
            `
        },
        {
            question: "Word Break",
            description: "Given a string s and a dictionary of words dict, determine if s can be segmented into words from dict.",
            hint: "Use dynamic programming to build solutions for substrings of s.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>We use a DP array where dp[i] represents whether s[0..i-1] can be segmented.</p>

                <pre><code>function wordBreak(s, wordDict) {
        const wordSet = new Set(wordDict);
        const dp = new Array(s.length + 1).fill(false);
        dp[0] = true;

        for (let i = 1; i <= s.length; i++) {
            for (let j = 0; j < i; j++) {
                if (dp[j] && wordSet.has(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[s.length];
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n²)</p>
                <p><strong>Space Complexity:</strong> O(n)</p>
            `
        },
        {
            question: "Course Schedule",
            description: "Determine if you can finish all courses given prerequisites represented as pairs [a, b] meaning you must take b before a.",
            hint: "This is a cycle detection problem in a directed graph. Use topological sorting.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>We model courses as a graph and perform topological sorting using Kahn's algorithm (BFS with in-degree counting).</p>

                <pre><code>function canFinish(numCourses, prerequisites) {
        const adj = Array.from({ length: numCourses }, () => []);
        const inDegree = new Array(numCourses).fill(0);

        for (const [course, prereq] of prerequisites) {
            adj[prereq].push(course);
            inDegree[course]++;
        }

        const queue = [];
        for (let i = 0; i < numCourses; i++) {
            if (inDegree[i] === 0) queue.push(i);
        }

        let count = 0;

        while (queue.length) {
            const current = queue.shift();
            count++;

            for (const neighbor of adj[current]) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] === 0) {
                    queue.push(neighbor);
                }
            }
        }

        return count === numCourses;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(V + E)</p>
                <p><strong>Space Complexity:</strong> O(V + E)</p>
            `
        },
        {
            question: "Product of Array Except Self",
            description: "Given an array nums, return an array where each element is the product of all elements except nums[i].",
            hint: "Use prefix and suffix products to compute the result in O(n) time without division.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>Compute prefix products in one pass, then multiply with suffix products in a second pass.</p>

                <pre><code>function productExceptSelf(nums) {
        const n = nums.length;
        const result = new Array(n);

        // Compute prefix products
        result[0] = 1;
        for (let i = 1; i < n; i++) {
            result[i] = result[i - 1] * nums[i - 1];
        }

        // Multiply with suffix products
        let suffix = 1;
        for (let i = n - 1; i >= 0; i--) {
            result[i] *= suffix;
            suffix *= nums[i];
        }

        return result;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n)</p>
                <p><strong>Space Complexity:</strong> O(1) (excluding output array)</p>
            `
        },
        {
            question: "Search in Rotated Sorted Array",
            description: "Given a rotated sorted array and a target value, return the index of target in O(log n) time.",
            hint: "Modified binary search to determine which side is properly sorted.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>Perform binary search while checking which half is properly sorted to determine where to search next.</p>

                <pre><code>function search(nums, target) {
        let left = 0, right = nums.length - 1;

        while (left <= right) {
            const mid = Math.floor((left + right) / 2);

            if (nums[mid] === target) return mid;

            // Left half is sorted
            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            // Right half is sorted
            else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }

        return -1;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(log n)</p>
                <p><strong>Space Complexity:</strong> O(1)</p>
            `
        },
        {
            question: "3Sum",
            description: "Given an array nums, return all unique triplets that sum to zero.",
            hint: "Sort the array first, then use two pointers for each element.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>Sort the array, then for each element, use two pointers to find pairs that sum to the complement.</p>

                <pre><code>function threeSum(nums) {
        nums.sort((a, b) => a - b);
        const result = [];

        for (let i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] === nums[i - 1]) continue;

            let left = i + 1, right = nums.length - 1;

            while (left < right) {
                const sum = nums[i] + nums[left] + nums[right];

                if (sum === 0) {
                    result.push([nums[i], nums[left], nums[right]]);
                    while (left < right && nums[left] === nums[left + 1]) left++;
                    while (left < right && nums[right] === nums[right - 1]) right--;
                    left++;
                    right--;
                } else if (sum < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }

        return result;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n²)</p>
                <p><strong>Space Complexity:</strong> O(1) (excluding output)</p>
            `
        },
        {
            question: "Clone Graph",
            description: "Given a reference of a node in a connected undirected graph, return a deep copy of the graph.",
            hint: "Use DFS or BFS with a hash map to store cloned nodes.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>We use BFS to traverse the graph while maintaining a map of original nodes to their clones.</p>

                <pre><code>function cloneGraph(node) {
        if (!node) return null;

        const visited = new Map();
        const queue = [node];
        visited.set(node, new Node(node.val));

        while (queue.length) {
            const current = queue.shift();

            for (const neighbor of current.neighbors) {
                if (!visited.has(neighbor)) {
                    visited.set(neighbor, new Node(neighbor.val));
                    queue.push(neighbor);
                }
                visited.get(current).neighbors.push(visited.get(neighbor));
            }
        }

        return visited.get(node);
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(V + E)</p>
                <p><strong>Space Complexity:</strong> O(V)</p>
            `
        },
        {
            question: "Word Ladder",
            description: "Given two words (beginWord and endWord) and a dictionary, find the length of shortest transformation sequence.",
            hint: "Use BFS treating each word as a node and one-letter transformations as edges.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>BFS where we generate all possible one-letter transformations and check if they're in the dictionary.</p>

                <pre><code>function ladderLength(beginWord, endWord, wordList) {
        const wordSet = new Set(wordList);
        if (!wordSet.has(endWord)) return 0;

        const queue = [[beginWord, 1]];
        const visited = new Set([beginWord]);

        while (queue.length) {
            const [word, level] = queue.shift();

            if (word === endWord) return level;

            for (let i = 0; i < word.length; i++) {
                for (let c = 97; c <= 122; c++) {
                    const newWord = word.slice(0, i) + String.fromCharCode(c) + word.slice(i + 1);

                    if (wordSet.has(newWord) && !visited.has(newWord)) {
                        queue.push([newWord, level + 1]);
                        visited.add(newWord);
                    }
                }
            }
        }

        return 0;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(M²×N) where M is word length and N is dictionary size</p>
                <p><strong>Space Complexity:</strong> O(N)</p>
            `
        },
        {
            question: "Median of Two Sorted Arrays",
            description: "Given two sorted arrays nums1 and nums2, return the median of the two sorted arrays.",
            hint: "Perform a modified binary search on the smaller array to find the correct partition.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>We perform binary search on the smaller array. For each partition, we compare the maximum element of the left part of both arrays with the minimum element of the right part. Based on this comparison, we adjust the partition to find the median.</p>

                <pre><code>function findMedianSortedArrays(nums1, nums2) {
        if (nums1.length > nums2.length) {
            return findMedianSortedArrays(nums2, nums1); // Ensure nums1 is the smaller array
        }

        const m = nums1.length;
        const n = nums2.length;
        const half = Math.floor((m + n + 1) / 2);

        let left = 0;
        let right = m;

        while (left <= right) {
            const partitionA = Math.floor((left + right) / 2);
            const partitionB = half - partitionA;

            const maxLeftA = partitionA === 0 ? -Infinity : nums1[partitionA - 1];
            const minRightA = partitionA === m ? Infinity : nums1[partitionA];

            const maxLeftB = partitionB === 0 ? -Infinity : nums2[partitionB - 1];
            const minRightB = partitionB === n ? Infinity : nums2[partitionB];

            if (maxLeftA <= minRightB && maxLeftB <= minRightA) {
                if ((m + n) % 2 === 0) {
                    return (Math.max(maxLeftA, maxLeftB) + Math.min(minRightA, minRightB)) / 2;
                } else {
                    return Math.max(maxLeftA, maxLeftB);
                }
            } else if (maxLeftA > minRightB) {
                right = partitionA - 1;
            } else {
                left = partitionA + 1;
            }
        }
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(log(min(m, n)))</p>
                <p><strong>Space Complexity:</strong> O(1)</p>
            `
        },
        {
            question: "Longest Palindromic Substring",
            description: "Given a string s, find the longest palindromic substring in s.",
            hint: "Use dynamic programming or the expand around center approach.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>We can use the expand around center approach. Iterate through each character and consider it as the center of a potential palindrome (both odd and even length).</p>

                <pre><code>function longestPalindrome(s) {
        if (!s) return "";

        let start = 0;
        let end = 0;

        function expandAroundCenter(left, right) {
            while (left >= 0 && right < s.length && s[left] === s[right]) {
                if (right - left > end - start) {
                    start = left;
                    end = right;
                }
                left--;
                right++;
            }
        }

        for (let i = 0; i < s.length; i++) {
            expandAroundCenter(i, i); // Odd length palindrome
            expandAroundCenter(i, i + 1); // Even length palindrome
        }

        return s.substring(start, end + 1);
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n²)</p>
                <p><strong>Space Complexity:</strong> O(1)</p>
            `
        },
        {
            question: "Reverse Linked List",
            description: "Given the head of a singly linked list, reverse the list.",
            hint: "Iterate through the list and change the next pointer of each node.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>Maintain pointers to the current node, previous node, and next node. In each step, reverse the current node's next pointer.</p>

                <pre><code>function reverseList(head) {
        let prev = null;
        let current = head;
        while (current) {
            const next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        return prev;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n)</p>
                <p><strong>Space Complexity:</strong> O(1)</p>
            `
        },
        {
            question: "Merge Two Sorted Lists",
            description: "You are given the heads of two sorted linked lists list1 and list2. Merge the two lists into one sorted list.",
            hint: "Iterate through both lists and compare nodes to build the merged list.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>Use a dummy head node and compare the heads of the two lists. Append the smaller node to the merged list and advance the corresponding pointer.</p>

                <pre><code>function mergeTwoLists(list1, list2) {
        const dummyHead = new ListNode(0);
        let current = dummyHead;

        while (list1 && list2) {
            if (list1.val < list2.val) {
                current.next = list1;
                list1 = list1.next;
            } else {
                current.next = list2;
                list2 = list2.next;
            }
            current = current.next;
        }

        if (list1) current.next = list1;
        if (list2) current.next = list2;

        return dummyHead.next;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(m + n)</p>
                <p><strong>Space Complexity:</strong> O(1)</p>
            `
        },
        {
            question: "Binary Tree Inorder Traversal",
            description: "Given the root of a binary tree, return the inorder traversal of its nodes' values.",
            hint: "Use recursion or an iterative approach with a stack.",
            answer: `
                <p><strong>Solution Approach (Iterative):</strong></p>
                <p>Use a stack to keep track of nodes. Go left as far as possible, then process the node, and then go right.</p>

                <pre><code>function inorderTraversal(root) {
        const result = [];
        const stack = [];
        let current = root;

        while (current || stack.length > 0) {
            while (current) {
                stack.push(current);
                current = current.left;
            }
            current = stack.pop();
            result.push(current.val);
            current = current.right;
        }

        return result;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n)</p>
                <p><strong>Space Complexity:</strong> O(n)</p>
            `
        },
        {
            question: "Binary Tree Level Order Traversal",
            description: "Given the root of a binary tree, return the level order traversal of its nodes' values.",
            hint: "Use Breadth-First Search (BFS) with a queue.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>Use a queue to process nodes level by level. For each level, process all nodes at that level before moving to the next.</p>

                <pre><code>function levelOrder(root) {
        if (!root) return [];
        const result = [];
        const queue = [root];

        while (queue.length > 0) {
            const levelSize = queue.length;
            const currentLevel = [];
            for (let i = 0; i < levelSize; i++) {
                const node = queue.shift();
                currentLevel.push(node.val);
                if (node.left) queue.push(node.left);
                if (node.right) queue.push(node.right);
            }
            result.push(currentLevel);
        }
        return result;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n)</p>
                <p><strong>Space Complexity:</strong> O(n)</p>
            `
        },
        {
            question: "Maximum Depth of Binary Tree",
            description: "Given the root of a binary tree, return its maximum depth.",
            hint: "Use recursion. The depth of a node is 1 + the maximum depth of its subtrees.",
            answer: `
                <p><strong>Solution Approach (Recursive):</strong></p>
                <p>The base case is when the node is null, in which case the depth is 0. Otherwise, the depth is 1 plus the maximum of the depths of the left and right subtrees.</p>

                <pre><code>function maxDepth(root) {
        if (!root) {
            return 0;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n)</p>
                <p><strong>Space Complexity:</strong> O(n) in the worst case (skewed tree) due to recursion stack.</p>
            `
        },
        {
            question: "Validate Binary Search Tree",
            description: "Given the root of a binary tree, determine if it is a valid binary search tree (BST).",
            hint: "Use recursion with a helper function that tracks the valid range for each node.",
            answer: `
                <p><strong>Solution Approach (Recursive with Helper):</strong></p>
                <p>For each node, we need to ensure its value is within a valid range defined by its ancestors.</p>

                <pre><code>function isValidBST(root) {
        return isValidBSTHelper(root, null, null);
    }

    function isValidBSTHelper(node, min, max) {
        if (!node) {
            return true;
        }
        if ((min !== null && node.val <= min) || (max !== null && node.val >= max)) {
            return false;
        }
        return isValidBSTHelper(node.left, min, node.val) && isValidBSTHelper(node.right, node.val, max);
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n)</p>
                <p><strong>Space Complexity:</strong> O(n) in the worst case (skewed tree) due to recursion stack.</p>
            `
        },
        {
            question: "Lowest Common Ancestor of a Binary Tree",
            description: "Given a binary tree and two nodes p and q, find the lowest common ancestor (LCA) of p and q.",
            hint: "Use recursion. If the current node is either p or q, or if p and q are in different subtrees, the current node is the LCA.",
            answer: `
                <p><strong>Solution Approach (Recursive):</strong></p>
                <p>Recursively search the left and right subtrees for p and q. If one subtree contains both, the LCA is in that subtree. If each subtree contains one, the current node is the LCA.</p>

                <pre><code>function lowestCommonAncestor(root, p, q) {
        if (!root || root === p || root === q) {
            return root;
        }

        const leftLCA = lowestCommonAncestor(root.left, p, q);
        const rightLCA = lowestCommonAncestor(root.right, p, q);

        if (leftLCA && rightLCA) {
            return root;
        }

        return leftLCA ? leftLCA : rightLCA;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n)</p>
                <p><strong>Space Complexity:</strong> O(n) in the worst case (skewed tree) due to recursion stack.</p>
            `
        },
        {
            question: "Implement Trie (Prefix Tree)",
            description: "Implement a trie with insert, search, and startsWith methods.",
            hint: "Use a tree-like structure where each node represents a character.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>Each node in the trie will store a map of children (characters to nodes) and a boolean flag indicating if the node represents the end of a word.</p>

                <pre><code>class TrieNode {
        constructor() {
            this.children = new Map();
            this.isEndOfWord = false;
        }
    }

    class Trie {
        constructor() {
            this.root = new TrieNode();
        }

        insert(word) {
            let current = this.root;
            for (const char of word) {
                if (!current.children.has(char)) {
                    current.children.set(char, new TrieNode());
                }
                current = current.children.get(char);
            }
            current.isEndOfWord = true;
        }

        search(word) {
            let current = this.root;
            for (const char of word) {
                if (!current.children.has(char)) {
                    return false;
                }
                current = current.children.get(char);
            }
            return current.isEndOfWord;
        }

        startsWith(prefix) {
            let current = this.root;
            for (const char of prefix) {
                if (!current.children.has(char)) {
                    return false;
                }
                current = current.children.get(char);
            }
            return true;
        }
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(L) for insert, search, and startsWith, where L is the length of the word/prefix.</p>
                <p><strong>Space Complexity:</strong> O(T), where T is the total number of characters in all inserted words.</p>
            `
        },
        {
            question: "Add Two Numbers",
            description: "You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.",
            hint: "Iterate through both lists, adding digits and carrying over the remainder.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>Iterate through both linked lists simultaneously. Add the corresponding digits along with any carry from the previous step. Create a new node for the sum's digit and update the carry.</p>

                <pre><code>function addTwoNumbers(l1, l2) {
        const dummyHead = new ListNode(0);
        let current = dummyHead;
        let carry = 0;

        while (l1 || l2 || carry) {
            const sum = (l1 ? l1.val : 0) + (l2 ? l2.val : 0) + carry;
            const digit = sum % 10;
            carry = Math.floor(sum / 10);

            current.next = new ListNode(digit);
            current = current.next;

            if (l1) l1 = l1.next;
            if (l2) l2 = l2.next;
        }

        return dummyHead.next;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(max(m, n)), where m and n are the lengths of the two linked lists.</p>
                <p><strong>Space Complexity:</strong> O(max(m, n) + 1) for the result list.</p>
            `
        },
        {
            question: "Longest Substring Without Repeating Characters",
            description: "Given a string s, find the length of the longest substring without repeating characters.",
            hint: "Use a sliding window approach with a hash map to track character frequencies.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>Maintain a sliding window [left, right] and a hash map to store the frequency of characters in the current window. Expand the window to the right. If a repeating character is found, shrink the window from the left until the repetition is removed.</p>

                <pre><code>function lengthOfLongestSubstring(s) {
        let left = 0;
        let right = 0;
        let maxLength = 0;
        const charMap = new Map();

        while (right < s.length) {
            const currentChar = s[right];
            charMap.set(currentChar, (charMap.get(currentChar) || 0) + 1);

            while (charMap.get(currentChar) > 1) {
                const leftChar = s[left];
                charMap.set(leftChar, charMap.get(leftChar) - 1);
                left++;
            }

            maxLength = Math.max(maxLength, right - left + 1);
            right++;
        }

        return maxLength;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n)</p>
                <p><strong>Space Complexity:</strong> O(min(m, n)), where n is the length of the string and m is the size of the character set.</p>
            `
        },
        {
            question: "String to Integer (atoi)",
            description: "Implement the atoi function which converts a string to an integer.",
            hint: "Handle leading whitespaces, signs, digits, and invalid input.",
            answer: `
                <p><strong>Solution Approach:</strong></p>
                <p>Process the string character by character, handling leading whitespaces, an optional sign, and then converting digits to an integer. Stop when a non-digit character is encountered or the integer overflows.</p>

                <pre><code>function myAtoi(s) {
        let i = 0;
        let sign = 1;
        let result = 0;
        const INT_MAX = 2147483647;
        const INT_MIN = -2147483648;

        // Skip leading whitespaces
        while (i < s.length && s[i] === ' ') {
            i++;
        }

        // Handle sign
        if (i < s.length && (s[i] === '+' || s[i] === '-')) {
            sign = s[i] === '+' ? 1 : -1;
            i++;
        }

        // Convert digits to integer
        while (i < s.length && /\d/.test(s[i])) {
            const digit = parseInt(s[i]);
            // Check for overflow before updating result
            if (result > Math.floor(INT_MAX / 10) || (result === Math.floor(INT_MAX / 10) && digit > 7)) {
                return sign === 1 ? INT_MAX : INT_MIN;
            }
            if (result < Math.floor(INT_MIN / 10) || (result === Math.floor(INT_MIN / 10) && digit < -8)) {
                return sign === 1 ? INT_MAX : INT_MIN;
            }
            result = result * 10 + digit;
            i++;
        }

        return result * sign;
    }</code></pre>

                <p><strong>Time Complexity:</strong> O(n), where n is the length of the string.</p>
                <p><strong>Space Complexity:</strong> O(1)</p>
            `
        }
    ];

    fasalQuestions.forEach((question, index) => {
        const questionDiv = document.createElement('div');
        questionDiv.classList.add('question-item');

        const title = document.createElement('h3');
        title.textContent = `${index + 1}. ${question.question}`;

        const description = document.createElement('p');
        description.textContent = question.description;

        // Create button container
        const buttonContainer = document.createElement('div');
        buttonContainer.style.display = 'flex';
        buttonContainer.style.gap = '10px';
        buttonContainer.style.marginTop = '15px';

        // Hint Button
        const hintButton = document.createElement('button');
        hintButton.textContent = 'Show Hint';
        hintButton.style.padding = '10px 20px';
        hintButton.style.border = 'none';
        hintButton.style.borderRadius = '5px';
        hintButton.style.backgroundColor = '#4CAF50';
        hintButton.style.color = 'white';
        hintButton.style.fontWeight = 'bold';
        hintButton.style.cursor = 'pointer';
        hintButton.style.transition = 'all 0.3s ease';
        hintButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        
        // Hover effect for hint button
        hintButton.addEventListener('mouseover', () => {
            hintButton.style.backgroundColor = '#45a049';
            hintButton.style.transform = 'translateY(-2px)';
            hintButton.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
        });
        
        hintButton.addEventListener('mouseout', () => {
            hintButton.style.backgroundColor = '#4CAF50';
            hintButton.style.transform = 'translateY(0)';
            hintButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        });
        
        hintButton.addEventListener('click', () => {
            hintQuestionTitle.textContent = question.question;
            hintContentDiv.innerHTML = `<p>${question.hint}</p>`;
            hintModal.style.display = 'block';
        });

        // Answer Button
        const answerButton = document.createElement('button');
        answerButton.textContent = 'Show Answer';
        answerButton.style.padding = '10px 20px';
        answerButton.style.border = 'none';
        answerButton.style.borderRadius = '5px';
        answerButton.style.backgroundColor = '#2196F3';
        answerButton.style.color = 'white';
        answerButton.style.fontWeight = 'bold';
        answerButton.style.cursor = 'pointer';
        answerButton.style.transition = 'all 0.3s ease';
        answerButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        
        // Hover effect for answer button
        answerButton.addEventListener('mouseover', () => {
            answerButton.style.backgroundColor = '#0b7dda';
            answerButton.style.transform = 'translateY(-2px)';
            answerButton.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
        });
        
        answerButton.addEventListener('mouseout', () => {
            answerButton.style.backgroundColor = '#2196F3';
            answerButton.style.transform = 'translateY(0)';
            answerButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        });
        
        answerButton.addEventListener('click', () => {
            hintQuestionTitle.textContent = question.question;
            hintContentDiv.innerHTML = question.answer;
            hintModal.style.display = 'block';
        });

        // Add buttons to container
        buttonContainer.appendChild(hintButton);
        buttonContainer.appendChild(answerButton);

        questionDiv.appendChild(title);
        questionDiv.appendChild(description);
        questionDiv.appendChild(buttonContainer);
        questionsListDiv.appendChild(questionDiv);
    });

    closeBtn.addEventListener('click', () => {
        hintModal.style.display = 'none';
    });

    window.addEventListener('click', (event) => {
        if (event.target === hintModal) {
            hintModal.style.display = 'none';
        }
    });
});