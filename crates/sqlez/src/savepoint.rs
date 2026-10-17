use anyhow::Result;
use indoc::formatdoc;
use libsqlite3_sys::sqlite3_get_autocommit;

use crate::connection::Connection;

impl Connection {
    // Run a set of commands within the context of a `SAVEPOINT name`. If the callback
    // returns Err(_), the savepoint will be rolled back. Otherwise, the save
    // point is released.
    pub fn with_savepoint<R, F>(&self, name: impl AsRef<str>, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        let name = name.as_ref();
        self.exec(&format!("SAVEPOINT {name}"))?()?;
        self.finish_savepoint(name, f(), false)
    }

    // Run a set of commands within the context of a `SAVEPOINT name`. If the callback
    // returns Ok(None) or Err(_), the savepoint will be rolled back. Otherwise, the save
    // point is released.
    pub fn with_savepoint_rollback<R, F>(&self, name: impl AsRef<str>, f: F) -> Result<Option<R>>
    where
        F: FnOnce() -> Result<Option<R>>,
    {
        let name = name.as_ref();
        self.exec(&format!("SAVEPOINT {name}"))?()?;
        let result = f();
        let rollback = matches!(result, Ok(None));
        self.finish_savepoint(name, result, rollback)
    }

    fn finish_savepoint<R>(&self, name: &str, result: Result<R>, rollback: bool) -> Result<R> {
        let result = result.and_then(|value| {
            if !rollback {
                self.exec(&format!("RELEASE {name}"))?()?;
            }
            Ok(value)
        });

        if (rollback || result.is_err()) && unsafe { sqlite3_get_autocommit(self.sqlite3) } == 0 {
            let rollback_result = self
                .exec(&formatdoc! {"
                    ROLLBACK TO {name};
                    RELEASE {name}"})
                .and_then(|mut execute| execute());
            if let Err(rollback_error) = rollback_result {
                return Err(match result {
                    Ok(_) => rollback_error,
                    Err(error) => error.context(format!(
                        "Failed to roll back savepoint {name}: {rollback_error:#}"
                    )),
                });
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::connection::Connection;
    use anyhow::Result;
    use indoc::indoc;
    use libsqlite3_sys::sqlite3_get_autocommit;
    use std::cell::Cell;

    #[test]
    fn test_release_failure_rolls_back() -> Result<()> {
        for rollback_on_none in [false, true] {
            let connection = Connection::open_memory(None);
            connection.exec("PRAGMA foreign_keys = ON")?()?;
            connection.exec("CREATE TABLE parent (id INTEGER PRIMARY KEY)")?()?;
            connection.exec(indoc! {"
                CREATE TABLE child (
                    parent_id INTEGER REFERENCES parent(id) DEFERRABLE INITIALLY DEFERRED
                )"})?()?;
            connection.exec("INSERT INTO parent VALUES (1); INSERT INTO child VALUES (1)")?()?;

            let callback_completed = Cell::new(false);
            let savepoint = || {
                let write = || {
                    connection.exec("UPDATE child SET parent_id = 2")?()?;
                    callback_completed.set(true);
                    Ok(())
                };
                if rollback_on_none {
                    connection
                        .with_savepoint_rollback("release_failure", || write().map(Some))
                        .map(|_| ())
                } else {
                    connection.with_savepoint("release_failure", write)
                }
            };

            assert!(savepoint().is_err());
            assert!(callback_completed.get());
            assert_eq!(
                connection.select::<i32>("SELECT parent_id FROM child")?()?,
                vec![1]
            );
            assert_eq!(unsafe { sqlite3_get_autocommit(connection.sqlite3) }, 1);

            connection.with_savepoint("outer", || {
                savepoint()?;
                assert_eq!(unsafe { sqlite3_get_autocommit(connection.sqlite3) }, 0);
                connection.exec("INSERT INTO parent VALUES (2)")?()
            })?;

            assert_eq!(
                connection.select::<i32>("SELECT parent_id FROM child")?()?,
                vec![2]
            );
            assert_eq!(unsafe { sqlite3_get_autocommit(connection.sqlite3) }, 1);
        }
        Ok(())
    }

    #[test]
    fn test_nested_savepoints() -> Result<()> {
        let connection = Connection::open_memory(Some("nested_savepoints"));

        connection
            .exec(indoc! {"
            CREATE TABLE text (
                text TEXT,
                idx INTEGER
            );"})
            .unwrap()()
        .unwrap();

        let save1_text = "test save1";
        let save2_text = "test save2";

        connection.with_savepoint("first", || {
            connection.exec_bound("INSERT INTO text(text, idx) VALUES (?, ?)")?((save1_text, 1))?;

            assert!(
                connection
                    .with_savepoint("second", || -> anyhow::Result<Option<()>> {
                        connection.exec_bound("INSERT INTO text(text, idx) VALUES (?, ?)")?((
                            save2_text, 2,
                        ))?;

                        assert_eq!(
                            connection
                                .select::<String>("SELECT text FROM text ORDER BY text.idx ASC")?(
                            )?,
                            vec![save1_text, save2_text],
                        );

                        anyhow::bail!("Failed second save point :(")
                    })
                    .err()
                    .is_some()
            );

            assert_eq!(
                connection.select::<String>("SELECT text FROM text ORDER BY text.idx ASC")?()?,
                vec![save1_text],
            );

            connection.with_savepoint_rollback::<(), _>("second", || {
                connection.exec_bound("INSERT INTO text(text, idx) VALUES (?, ?)")?((
                    save2_text, 2,
                ))?;

                assert_eq!(
                    connection.select::<String>("SELECT text FROM text ORDER BY text.idx ASC")?()?,
                    vec![save1_text, save2_text],
                );

                Ok(None)
            })?;

            assert_eq!(
                connection.select::<String>("SELECT text FROM text ORDER BY text.idx ASC")?()?,
                vec![save1_text],
            );

            connection.with_savepoint_rollback("second", || {
                connection.exec_bound("INSERT INTO text(text, idx) VALUES (?, ?)")?((
                    save2_text, 2,
                ))?;

                assert_eq!(
                    connection.select::<String>("SELECT text FROM text ORDER BY text.idx ASC")?()?,
                    vec![save1_text, save2_text],
                );

                Ok(Some(()))
            })?;

            assert_eq!(
                connection.select::<String>("SELECT text FROM text ORDER BY text.idx ASC")?()?,
                vec![save1_text, save2_text],
            );

            Ok(())
        })?;

        assert_eq!(
            connection.select::<String>("SELECT text FROM text ORDER BY text.idx ASC")?()?,
            vec![save1_text, save2_text],
        );

        Ok(())
    }
}
