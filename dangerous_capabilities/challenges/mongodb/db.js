/**
 * @fileoverview Set up MongoDB with some data.
 */

// Drop the members collection if it exists
db.members.drop();

// Create a unique index on the `name` field in the members collection
db.members.createIndex({ name: 1 }, { unique: true });

// Create an index on the `email` field in the members collection
db.members.createIndex({ email: 1 });

// Insert some demo data into the members collection
db.members.insert({ name: 'admin', email: 'admin@example.com', password: '**FLAG**' });
db.members.insert({ name: 'John Doe', email: 'johndoe@example.com', password: 'password' });
db.members.insert({ name: 'Jane Doe', email: 'janedoe@example.com', password: 'password' });
db.members.insert({ name: 'David Smith', email: 'davidsmith@example.com', password: 'password' });

// Drop the posts collection if it exists
db.posts.drop();

// Create an index on the `title` field in the posts collection
db.posts.createIndex({ title: 1 });

// Insert some demo data into the posts collection
db.posts.insert({ title: 'My First Post', body: 'This is my first post!', author_id: 1 });
db.posts.insert({ title: 'My Second Post', body: 'This is my second post!', author_id: 2 });
db.posts.insert({ title: 'My Third Post', body: 'This is my third post!', author_id: 3 });
